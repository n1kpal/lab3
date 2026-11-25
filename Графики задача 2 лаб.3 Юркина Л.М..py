from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # рисуем в файлы
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal as sps

# Константы

NOISE_TYPE = "airport"
SNR_LEVELS = (0, 5, 10, 15)                 # дБ, номинальные
SELECTED_IDS = ("sp03", "sp14", "sp26")

Signal = np.ndarray

DPI = 150
SPEC_NPERSEG = 256
SPEC_NOVERLAP = 192
SPEC_RANGE_DB = 80.0
FFT_RANGE_DB = 100.0
EPS = 1e-12

# Ключи словаря сигналов строго в таком порядке
KEYS = ("clean",) + tuple(f"{s}dB" for s in SNR_LEVELS)
PANEL_TITLES = {"clean": "Чистый эталон"}
PANEL_TITLES.update(
    {f"{s}dB": f"Шум «{NOISE_TYPE}», SNR {s} дБ (номинал)" for s in SNR_LEVELS}
)


#Чтение файлов
def _read_mono(path: Path) -> tuple[Signal, int]:
    x, fs = sf.read(path, dtype="float64")
    if x.ndim > 1:
        logging.warning("%s: %d каналов, берём 1-й", path.name, x.shape[1])
        x = x[:, 0]
    return x, int(fs)


def _load_pair(sp_id: str, snr_nominal_db: int, dataset_dir: Path) -> tuple[Signal, Signal, int]:
    #Возвращает (clean, noisy, fs).
    clean_path = dataset_dir / "clean" / f"{sp_id}.wav"
    folder = dataset_dir / "noisy" / NOISE_TYPE / f"{snr_nominal_db}dB"
    found = sorted(folder.glob(f"*{sp_id}*.wav"))
    if not clean_path.is_file():
        raise FileNotFoundError(f"Нет файла: {clean_path}")
    if len(found) != 1:
        raise FileNotFoundError(f"В {folder} для {sp_id} найдено файлов: {len(found)}")
    clean, fs_c = _read_mono(clean_path)
    noisy, fs_n = _read_mono(found[0])
    if fs_c != fs_n:
        raise ValueError(f"{sp_id}/{snr_nominal_db} дБ: разный fs ({fs_c} и {fs_n})")
    if len(clean) != len(noisy):
        raise ValueError(f"{sp_id}/{snr_nominal_db} дБ: разная длина ({len(clean)} и {len(noisy)})")
    return clean, noisy, fs_c


#Расчеты
def compute_fft_spectrum(x: Signal, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Односторонний rfft, окно Hann, нормировка по сумме окна.
    Возвращает (freqs_hz, mag_db), mag_db = 20*log10(|X| + 1e-12),
    дБ относительно полной шкалы (амплитуда 1.0 = 0 дБ).
    """
    x = np.asarray(x, dtype=np.float64)
    window = np.hanning(len(x))
    spectrum = np.fft.rfft(x * window) / np.sum(window)
    freqs_hz = np.fft.rfftfreq(len(x), d=1.0 / fs)
    mag_db = 20.0 * np.log10(np.abs(spectrum) + EPS)
    return freqs_hz, mag_db


def compute_spectrogram(
    x: Signal, fs: int, nperseg: int = SPEC_NPERSEG, noverlap: int = SPEC_NOVERLAP
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """scipy.signal.spectrogram, окно Hann. Возвращает (f_hz, t_s, s_db), дБ отн. полной шкалы."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < nperseg:  # защита для очень коротких сигналов
        nperseg = len(x)
        noverlap = nperseg * 3 // 4
    f_hz, t_s, mag = sps.spectrogram(
        x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
        mode="magnitude", scaling="spectrum",
    )
    return f_hz, t_s, 20.0 * np.log10(mag + EPS)


@dataclass
class PlotScales:
    amp_lim: tuple
    fft_db_lim: tuple
    spec_db_lim: tuple
    f_max_hz: float


def _check_signals(signals: dict[str, Signal]) -> None:
    if tuple(signals.keys()) != KEYS:
        raise ValueError(f"Ключи signals должны быть строго {KEYS}, получено {tuple(signals.keys())}")


def compute_common_scales(signals: dict[str, Signal], fs: int) -> PlotScales:
    #Единые шкалы для всех 5 сигналов одной фразы.
    _check_signals(signals)

    peak = max(float(np.max(np.abs(x))) for x in signals.values())
    amp = 1.05 * peak if peak > 0 else 1.0

    fft_max = fft_min = None
    spec_max = -np.inf
    for x in signals.values():
        _, mag_db = compute_fft_spectrum(x, fs)
        fft_max = mag_db.max() if fft_max is None else max(fft_max, mag_db.max())
        fft_min = mag_db.min() if fft_min is None else min(fft_min, mag_db.min())
        _, _, s_db = compute_spectrogram(x, fs)
        spec_max = max(spec_max, float(s_db.max()))

    fft_hi = np.ceil((fft_max + 1.0) / 10.0) * 10.0
    fft_lo = np.floor(max(fft_min, fft_hi - FFT_RANGE_DB) / 10.0) * 10.0  # не падаем в -240 дБ

    return PlotScales(
        amp_lim=(-amp, amp),
        fft_db_lim=(float(fft_lo), float(fft_hi)),
        spec_db_lim=(spec_max - SPEC_RANGE_DB, spec_max),  # vmin = vmax - 80 дБ
        f_max_hz=fs / 2.0,
    )



# Рисунки
def _save(fig, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def _line_color(key: str) -> str:
    return "tab:blue" if key == "clean" else "tab:red"


def plot_waveforms(
    signals: dict[str, Signal], fs: int, scales: PlotScales, title: str, out_path: Path
) -> Path:
    _check_signals(signals)
    t_max = max(len(x) for x in signals.values()) / fs
    fig, axes = plt.subplots(len(KEYS), 1, figsize=(10, 11), sharex=True, constrained_layout=True)
    for ax, key in zip(axes, KEYS):
        x = signals[key]
        ax.plot(np.arange(len(x)) / fs, x, linewidth=0.5, color=_line_color(key))
        ax.set_title(PANEL_TITLES[key], fontsize=10, loc="left")
        ax.set_ylim(*scales.amp_lim)
        ax.set_xlim(0, t_max)
        ax.set_xlabel("Время, с")
        ax.set_ylabel("Амплитуда (отн. ед., полная шкала = 1)", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{title}: осциллограмма (одинаковые пределы осей на всех панелях)")
    return _save(fig, out_path)


def plot_fft(
    signals: dict[str, Signal], fs: int, scales: PlotScales, title: str, out_path: Path
) -> Path:
    _check_signals(signals)
    fig, axes = plt.subplots(len(KEYS), 1, figsize=(10, 11), sharex=True, constrained_layout=True)
    for ax, key in zip(axes, KEYS):
        freqs, mag_db = compute_fft_spectrum(signals[key], fs)
        ax.plot(freqs, mag_db, linewidth=0.6, color=_line_color(key))
        ax.set_title(PANEL_TITLES[key], fontsize=10, loc="left")
        ax.set_xlim(0, scales.f_max_hz)
        ax.set_ylim(*scales.fft_db_lim)
        ax.set_xlabel("Частота, Гц")
        ax.set_ylabel("Уровень, дБ отн. полной шкалы", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{title}: спектр FFT (окно Hann, одинаковые шкалы)")
    return _save(fig, out_path)
def plot_spectrograms(
    signals: dict[str, Signal], fs: int, scales: PlotScales, title: str, out_path: Path
) -> Path:
    _check_signals(signals)
    t_max = max(len(x) for x in signals.values()) / fs
    vmin, vmax = scales.spec_db_lim
    fig, axes = plt.subplots(len(KEYS), 1, figsize=(10, 12), sharex=True, constrained_layout=True)
    mappable = None
    for ax, key in zip(axes, KEYS):
        f_hz, t_s, s_db = compute_spectrogram(signals[key], fs)
        mappable = ax.pcolormesh(t_s, f_hz, s_db, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(PANEL_TITLES[key], fontsize=10, loc="left")
        ax.set_xlim(0, t_max)
        ax.set_ylim(0, scales.f_max_hz)
        ax.set_xlabel("Время, с")
        ax.set_ylabel("Частота, Гц")
    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), pad=0.02)  # ОДНА шкала на все панели
    cbar.set_label("Уровень, дБ отн. полной шкалы")
    win_ms = 1000.0 * SPEC_NPERSEG / fs
    fig.suptitle(f"{title}: спектрограмма")
    fig.text(
        0.01, 0.005,
        f"Окно Hann, nperseg = {SPEC_NPERSEG} отсч. ({win_ms:.0f} мс при fs = {fs} Гц), "
        f"noverlap = {SPEC_NOVERLAP} отсч.; динамический диапазон {SPEC_RANGE_DB:.0f} дБ",
        fontsize=8, ha="left", va="bottom",
    )
    return _save(fig, out_path)


# Сборка
def make_stage2_figures(
    selected_ids: tuple[str, ...], dataset_dir: Path, out_dir: Path
) -> list[Path]:
    #Для каждой фразы: clean + 0/5/10/15 дБ -> 3 рисунка. В итоге 9 файлов.
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError(f"Фразы должны быть разными: {selected_ids}")
    if len(selected_ids) < 3:
        raise ValueError("По ТЗ нужно минимум три разные фразы")

    dataset_dir, out_dir = Path(dataset_dir), Path(out_dir)
    paths: list[Path] = []

    for sp_id in selected_ids:
        signals: dict[str, Signal] = {}
        fs_ref = None
        for snr in SNR_LEVELS:
            clean, noisy, fs = _load_pair(sp_id, snr, dataset_dir)
            if fs_ref is None:
                fs_ref = fs
                signals["clean"] = clean
            elif fs != fs_ref:
                raise ValueError(f"{sp_id}: fs различается между уровнями ({fs} и {fs_ref})")
            elif len(clean) != len(signals["clean"]):
                raise ValueError(f"{sp_id}: эталон разной длины в разных папках")
            signals[f"{snr}dB"] = noisy

        scales = compute_common_scales(signals, fs_ref)
        title = f"Фраза {sp_id}, шум «{NOISE_TYPE}»"
        paths.append(plot_waveforms(signals, fs_ref, scales, title, out_dir / f"stage2_{sp_id}_waveform.png"))
        paths.append(plot_fft(signals, fs_ref, scales, title, out_dir / f"stage2_{sp_id}_fft.png"))
        paths.append(plot_spectrograms(signals, fs_ref, scales, title, out_dir / f"stage2_{sp_id}_spectrogram.png"))
    return paths


def run_stage2(dataset_dir: Path, results_dir: Path) -> dict[str, Path]:
    #Единый вход этапа 2. Читает dataset_dir, пишет в results_dir/figures.
    paths = make_stage2_figures(SELECTED_IDS, Path(dataset_dir), Path(results_dir) / "figures")
    return {p.stem: p for p in paths}

#Запуск
def _main() -> int:
    global SELECTED_IDS
    parser = argparse.ArgumentParser(description="Этап 2: waveform, FFT, спектрограммы")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--selected", type=str, default=None, help="например: sp03,sp14,sp26")
    args = parser.parse_args()
    if args.selected:
        SELECTED_IDS = tuple(s.strip() for s in args.selected.split(",") if s.strip())
    try:
        result = run_stage2(args.dataset, args.results)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Ошибка: {exc}")
        return 2
    print(f"Готово: создано {len(result)} рисунков в {args.results / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())