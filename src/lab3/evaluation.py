from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

NOISE_TYPE = "airport"
SNR_LEVELS = (0, 5, 10, 15)
SELECTED_IDS = ("sp03", "sp14", "sp26")
DATASET_DIR = Path("dataset")
RESULTS_DIR = Path("results")

CLIP_THRESHOLD = 0.999
DPI = 150
SPEC_NPERSEG = 256
SPEC_NOVERLAP = 192
SPEC_RANGE_DB = 80.0
FFT_RANGE_DB = 100.0
EPS = 1e-12

FILTER = {
    "kind": "highpass",
    "family": "butter",
    "order": 4,
    "cutoff_hz": 100.0,
    "zero_phase": True,
}

KEYS = ("clean",) + tuple(f"{s}dB" for s in SNR_LEVELS)
SP_PATTERN = re.compile(r"(sp\d{2})", re.IGNORECASE)
SNR_PATTERN = re.compile(r"^([+-]?\d+)\s*dB$", re.IGNORECASE)

logger = logging.getLogger(__name__)
Signal = np.ndarray


class AlignmentError(ValueError):
    """Ошибка несовпадения длин или сдвига сигналов во времени."""

class Pair(NamedTuple):
    sp_id: str
    snr_nominal_db: int
    clean: Signal
    noisy: Signal
    fs: int

class AlignmentReport(NamedTuple):
    len_equal: bool
    lag_samples: int
    ls_gain: float
    corr: float

@dataclass
class PlotScales:
    amp_lim: tuple[float, float]
    fft_db_lim: tuple[float, float]
    spec_db_lim: tuple[float, float]
    f_max_hz: float


def _check_signal(x: Signal, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} должен быть непустым одномерным массивом")
    return arr


def rms(x: Signal) -> float:
    arr = _check_signal(x, "x")
    return float(np.sqrt(np.mean(arr * arr)))


def snr_db(clean: Signal, test: Signal) -> float:
    """Расчёт SNR строго по формуле ТЗ: 10 * log10(sum(clean^2) / sum((clean - test)^2))."""
    c = _check_signal(clean, "clean")
    t = _check_signal(test, "test")
    if c.shape != t.shape:
        raise ValueError(f"Разная длина сигналов: clean={c.size}, test={t.size}")

    sig_energy = float(np.sum(c * c))
    diff = c - t
    err_energy = float(np.sum(diff * diff))

    if err_energy == 0.0:
        if sig_energy == 0.0:
            raise ValueError("Энергия обоих сигналов равна нулю")
        return float("inf")
    if sig_energy == 0.0:
        return float("-inf")

    return float(10.0 * np.log10(sig_energy / err_energy))


def check_alignment(clean: Signal, test: Signal, max_lag: int = 400) -> AlignmentReport:
    """Проверка совпадения длин, лага взаимной корреляции и коэффициента масштаба ls_gain."""
    c = _check_signal(clean, "clean")
    t = _check_signal(test, "test")

    xcorr = signal.correlate(t, c, mode="full", method="fft")
    lags = signal.correlation_lags(t.size, c.size, mode="full")

    mask = np.abs(lags) <= max_lag
    valid_xcorr = xcorr[mask]
    valid_lags = lags[mask]
    best_index = int(np.argmax(valid_xcorr))
    lag_samples = int(valid_lags[best_index])

    denom = float(np.linalg.norm(c) * np.linalg.norm(t))
    corr = float(valid_xcorr[best_index] / denom) if denom > 0.0 else 0.0

    min_len = min(c.size, t.size)
    c_part, t_part = c[:min_len], t[:min_len]
    c_energy = float(np.sum(c_part * c_part))
    ls_gain = float(np.sum(c_part * t_part) / c_energy) if c_energy > 0.0 else 0.0

    return AlignmentReport(c.size == t.size, lag_samples, ls_gain, corr)


def load_wav(path: Path) -> tuple[Signal, int]:
    data, fs = sf.read(Path(path), dtype="float64", always_2d=False)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim > 1:
        logger.warning("В файле %s %d каналов — взят 1-й канал.", path, arr.shape[1])
        arr = arr[:, 0]
    return np.ascontiguousarray(arr, dtype=np.float64), int(fs)


def save_wav(path: Path, x: Signal, fs: int, dataset_dir: Path = DATASET_DIR) -> dict:
    out_path = Path(path)
    if out_path.resolve().is_relative_to(Path(dataset_dir).resolve()):
        raise ValueError(f"Запрещено писать внутрь dataset_dir: {out_path}")

    arr = _check_signal(x, "x")
    clip_info = check_clipping(arr, threshold=CLIP_THRESHOLD)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, arr, int(fs), subtype="PCM_16")
    return {"n_clipped": clip_info["n_clipped"], "peak": clip_info["peak"]}


def load_pair(sp_id: str, snr_nominal_db: int, dataset_dir: Path = DATASET_DIR) -> Pair:
    root = Path(dataset_dir)
    target_id = sp_id.lower()

    clean_hits = sorted((root / "clean").rglob(f"*{target_id}*.wav"))
    if len(clean_hits) != 1:
        raise FileNotFoundError(f"Для {target_id} найдено clean-файлов: {len(clean_hits)}")

    noisy_dir = root / "noisy" / NOISE_TYPE / f"{snr_nominal_db}dB"
    if not noisy_dir.exists():
        noisy_dir = root / "noisy"
    noisy_hits = [
        p for p in sorted(noisy_dir.rglob(f"*{target_id}*.wav"))
        if f"{snr_nominal_db}db" in p.as_posix().lower()
    ]
    if len(noisy_hits) != 1:
        raise FileNotFoundError(f"Для {target_id} ({snr_nominal_db} dB) найдено noisy-файлов: {len(noisy_hits)}")

    clean_sig, fs_c = load_wav(clean_hits[0])
    noisy_sig, fs_n = load_wav(noisy_hits[0])

    if fs_c != fs_n:
        raise ValueError(f"Разный fs для {target_id} ({snr_nominal_db} dB): {fs_c} != {fs_n}")
    if len(clean_sig) != len(noisy_sig):
        raise AlignmentError(f"Разная длина для {target_id} ({snr_nominal_db} dB): {len(clean_sig)} != {len(noisy_sig)}")

    return Pair(target_id, int(snr_nominal_db), clean_sig, noisy_sig, fs_c)


def compute_fft_spectrum(x: Signal, fs: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    window = np.hanning(len(x))
    spectrum = np.fft.rfft(x * window) / np.sum(window)
    freqs_hz = np.fft.rfftfreq(len(x), d=1.0 / fs)
    mag_db = 20.0 * np.log10(np.abs(spectrum) + EPS)
    return freqs_hz, mag_db


def compute_common_scales(signals: dict[str, Signal], fs: int) -> PlotScales:
    peak = max(float(np.max(np.abs(x))) for x in signals.values())
    amp = 1.05 * peak if peak > 0 else 1.0

    fft_max = fft_min = None
    spec_max = -np.inf
    for x in signals.values():
        _, mag_db = compute_fft_spectrum(x, fs)
        fft_max = mag_db.max() if fft_max is None else max(fft_max, mag_db.max())
        fft_min = mag_db.min() if fft_min is None else min(fft_min, mag_db.min())

        nperseg = min(SPEC_NPERSEG, len(x))
        noverlap = nperseg * 3 // 4
        _, _, mag = signal.spectrogram(
            x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
            mode="magnitude", scaling="spectrum",
        )
        s_db = 20.0 * np.log10(mag + EPS)
        spec_max = max(spec_max, float(s_db.max()))

    fft_hi = float(np.ceil((fft_max + 1.0) / 10.0) * 10.0)
    fft_lo = float(np.floor(max(fft_min, fft_hi - FFT_RANGE_DB) / 10.0) * 10.0)

    return PlotScales(
        amp_lim=(-amp, amp),
        fft_db_lim=(fft_lo, fft_hi),
        spec_db_lim=(spec_max - SPEC_RANGE_DB, spec_max),
        f_max_hz=fs / 2.0,
    )

def check_clipping(x: Signal, threshold: float = CLIP_THRESHOLD) -> dict:
    """Проверка клиппинга сигнала в памяти по порогу амплитуды."""
    arr = _check_signal(x, "x")
    abs_arr = np.abs(arr)
    n_clipped = int(np.sum(abs_arr >= threshold))
    peak = float(np.max(abs_arr))
    return {
        "n_clipped": n_clipped,
        "peak": peak,
        "has_clipping": n_clipped > 0,
    }


def check_saved_wav_clipping(path: Path) -> dict:
    """Проверка СОХРАНЁННОГО WAV-файла: в int16 крайние значения -32768 и 32767 — клиппинг."""
    wav_path = Path(path)
    if not wav_path.is_file():
        raise FileNotFoundError(f"WAV-файл не найден: {wav_path}")

    info = sf.info(wav_path)
    if info.subtype == "PCM_16":
        data_int, _ = sf.read(wav_path, dtype="int16", always_2d=False)
        arr = np.asarray(data_int)
        if arr.ndim > 1:
            arr = arr[:, 0]
        n_clipped = int(np.sum((arr == -32768) | (arr == 32767)))
        peak = float(np.max(np.abs(arr.astype(np.float64))) / 32768.0) if arr.size > 0 else 0.0
        return {
            "n_clipped": n_clipped,
            "peak": peak,
            "has_clipping": n_clipped > 0,
        }

    sig, _ = load_wav(wav_path)
    return check_clipping(sig, threshold=CLIP_THRESHOLD)


def evaluate_record(clean: Signal, noisy: Signal, processed: Signal, fs: int) -> dict:
    """Расчёт метрик SNR до/после, ΔSNR, RMS, пика, клиппинга и ls_gain."""
    c = _check_signal(clean, "clean")
    n = _check_signal(noisy, "noisy")
    p = _check_signal(processed, "processed")

    if not (len(p) == len(c) == len(n)):
        raise AlignmentError(f"Несовпадение длин: clean={len(c)}, noisy={len(n)}, processed={len(p)}")
    if fs <= 0:
        raise ValueError(f"Некорректный fs={fs}")

    align_n = check_alignment(c, n)
    align_p = check_alignment(c, p)
    if not align_n.len_equal or align_n.lag_samples != 0:
        raise AlignmentError(f"Сдвиг noisy относительно clean: lag={align_n.lag_samples}")
    if not align_p.len_equal or align_p.lag_samples != 0:
        raise AlignmentError(f"Сдвиг processed относительно clean: lag={align_p.lag_samples}")

    s_before = snr_db(c, n)
    s_after = snr_db(c, p)
    d_snr = float(s_after - s_before)

    clip_info = check_clipping(p, threshold=CLIP_THRESHOLD)
    return {
        "snr_before": float(s_before),
        "snr_after": float(s_after),
        "delta_snr": d_snr,
        "rms_before": rms(n),
        "rms_after": rms(p),
        "peak_after": clip_info["peak"],
        "n_clipped": clip_info["n_clipped"],
        "ls_gain_after": float(align_p.ls_gain),
    }


def auto_comment(row_before: pd.Series, row_after: pd.Series) -> str:
    """Черновик комментария по ΔSNR и клиппингу для дальнейшей ручной доработки."""
    d_snr = float(row_after["delta_snr"])
    n_clip = int(row_after.get("n_clipped", 0))

    if d_snr >= 1.0:
        msg = f"помог заметно (ΔSNR = +{d_snr:.2f} дБ)"
    elif d_snr >= 0.0:
        msg = f"слабое улучшение (ΔSNR = +{d_snr:.2f} дБ)"
    else:
        msg = f"ухудшил (ΔSNR = {d_snr:.2f} дБ)"

    if n_clip > 0:
        msg += f"; ВНИМАНИЕ: клиппинг ({n_clip} отсч.)"

    return f"{msg}. Причина: [дописать вручную]"


def _parse_filter_spec(spec: object) -> dict:
    if spec is None:
        spec = FILTER

    if hasattr(spec, "to_dict"):
        d = spec.to_dict()
        return {
            "filter_type": d.get("filter_type", ""),
            "filter_family": "butter",
            "filter_order": d.get("order", ""),
            "filter_cutoff_hz": d.get("cutoff_hz", ""),
            "filter_zero_phase": d.get("zero_phase", ""),
        }

    if isinstance(spec, dict):
        return {
            "filter_type": spec.get("kind", spec.get("filter_type", "")),
            "filter_family": spec.get("family", "butter"),
            "filter_order": spec.get("order", ""),
            "filter_cutoff_hz": spec.get("cutoff_hz", ""),
            "filter_zero_phase": spec.get("zero_phase", ""),
        }

    return {
        "filter_type": getattr(spec, "typ", getattr(spec, "kind", "")),
        "filter_family": getattr(spec, "family", "butter"),
        "filter_order": getattr(spec, "ord", getattr(spec, "order", "")),
        "filter_cutoff_hz": getattr(spec, "cut", getattr(spec, "cutoff_hz", "")),
        "filter_zero_phase": getattr(spec, "zph", getattr(spec, "zero_phase", "")),
    }


def build_experiment_table(
    records: pd.DataFrame,
    spec: object = None,
    dataset_dir: Path = DATASET_DIR,
) -> pd.DataFrame:
    """Формирует таблицу experiments.csv на 24 строки («до» и «после» на каждую из 12 записей)."""
    ddr = Path(dataset_dir)
    f_meta = _parse_filter_spec(spec)
    rows = []

    for _, rec in records.iterrows():
        sp_id = str(rec["sp_id"]).lower()
        snr_nom = int(rec["snr_nominal_db"])
        out_path = Path(rec["out_path"])

        pair = load_pair(sp_id, snr_nom, dataset_dir=ddr)
        proc_sig, proc_fs = load_wav(out_path)

        if proc_fs != pair.fs:
            raise ValueError(f"{sp_id} ({snr_nom} dB): fs в WAV ({proc_fs}) != fs датасета ({pair.fs})")

        metrics = evaluate_record(pair.clean, pair.noisy, proc_sig, pair.fs)
        wav_clip = check_saved_wav_clipping(out_path)
        noisy_clip = check_clipping(pair.noisy, threshold=CLIP_THRESHOLD)
        n_clipped_total = max(metrics["n_clipped"], wav_clip["n_clipped"])

        row_before = {
            "sp_id": sp_id,
            "noise_type": NOISE_TYPE,
            "snr_nominal_db": snr_nom,
            "stage": "before",
            "fs_hz": pair.fs,
            "n_samples": len(pair.noisy),
            "filter_type": np.nan,
            "filter_family": np.nan,
            "filter_order": np.nan,
            "filter_cutoff_hz": np.nan,
            "filter_zero_phase": np.nan,
            "snr_db": metrics["snr_before"],
            "delta_snr": np.nan,
            "rms": metrics["rms_before"],
            "peak": noisy_clip["peak"],
            "n_clipped": noisy_clip["n_clipped"],
            "ls_gain": float(check_alignment(pair.clean, pair.noisy).ls_gain),
            "wav_path": "",
            "comment": "исходный зашумлённый сигнал",
        }

        row_after = {
            "sp_id": sp_id,
            "noise_type": NOISE_TYPE,
            "snr_nominal_db": snr_nom,
            "stage": "after",
            "fs_hz": proc_fs,
            "n_samples": len(proc_sig),
            "filter_type": f_meta["filter_type"],
            "filter_family": f_meta["filter_family"],
            "filter_order": f_meta["filter_order"],
            "filter_cutoff_hz": f_meta["filter_cutoff_hz"],
            "filter_zero_phase": f_meta["filter_zero_phase"],
            "snr_db": metrics["snr_after"],
            "delta_snr": metrics["delta_snr"],
            "rms": metrics["rms_after"],
            "peak": max(metrics["peak_after"], wav_clip["peak"]),
            "n_clipped": n_clipped_total,
            "ls_gain": metrics["ls_gain_after"],
            "wav_path": str(out_path),
            "comment": "",
        }
        row_after["comment"] = auto_comment(pd.Series(row_before), pd.Series(row_after))

        rows.append(row_before)
        rows.append(row_after)

    df = pd.DataFrame(rows)
    if df["snr_db"].isna().any():
        raise RuntimeError("В таблице experiments есть пустые значения snr_db!")
    return df


def summarize_effect(exp: pd.DataFrame) -> dict:
    """Сводка по уровням SNR: средний ΔSNR, число улучшений/ухудшений, лучшая и худшая запись."""
    after_df = exp[exp["stage"] == "after"].copy()
    if after_df.empty:
        raise ValueError("Нет строк со stage == 'after'")

    mean_by_level = {}
    for lvl in SNR_LEVELS:
        sub = after_df[after_df["snr_nominal_db"] == lvl]["delta_snr"].dropna()
        mean_by_level[int(lvl)] = float(sub.mean()) if not sub.empty else float("nan")

    deltas = after_df["delta_snr"].astype(float)
    best_row = after_df.loc[deltas.idxmax()]
    worst_row = after_df.loc[deltas.idxmin()]

    return {
        "mean_delta_snr_by_level": mean_by_level,
        "overall_mean_delta_snr": float(deltas.mean()),
        "n_positive": int((deltas > 0).sum()),
        "n_negative": int((deltas < 0).sum()),
        "best_record": {
            "sp_id": str(best_row["sp_id"]),
            "snr_nominal_db": int(best_row["snr_nominal_db"]),
            "delta_snr": float(best_row["delta_snr"]),
        },
        "worst_record": {
            "sp_id": str(worst_row["sp_id"]),
            "snr_nominal_db": int(worst_row["snr_nominal_db"]),
            "delta_snr": float(worst_row["delta_snr"]),
        },
    }


def plot_before_after_spectra(
    sp_id: str,
    snr_nominal_db: int,
    dataset_dir: Path,
    out_path: Path,
    scales: PlotScales | None = None,
    processed_dir: Path | None = None,
) -> Path:
    """FFT-спектры clean, noisy и processed на одних осях и единых шкалах."""
    ddr = Path(dataset_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pair = load_pair(sp_id, snr_nominal_db, dataset_dir=ddr)

    if processed_dir is None:
        processed_dir = out_path.parent.parent / "audio_processed"
    proc_path = Path(processed_dir) / f"{sp_id}_{NOISE_TYPE}_{snr_nominal_db}dB_filtered.wav"
    if not proc_path.is_file():
        raise FileNotFoundError(f"Не найден обработанный WAV: {proc_path}")
    proc_sig, proc_fs = load_wav(proc_path)
    if proc_fs != pair.fs or len(proc_sig) != len(pair.clean):
        raise AlignmentError(f"Несовпадение fs или длины для {sp_id} ({snr_nominal_db} dB)")

    if scales is None:
        all_sigs = {"clean": pair.clean}
        for lvl in SNR_LEVELS:
            p_lvl = load_pair(sp_id, lvl, dataset_dir=ddr)
            all_sigs[f"{lvl}dB"] = p_lvl.noisy
        scales = compute_common_scales(all_sigs, pair.fs)

    f_c, db_c = compute_fft_spectrum(pair.clean, pair.fs)
    f_n, db_n = compute_fft_spectrum(pair.noisy, pair.fs)
    f_p, db_p = compute_fft_spectrum(proc_sig, pair.fs)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(f_n, db_n, color="tab:red", alpha=0.55, lw=0.8, label=f"Noisy ({snr_nominal_db} дБ)")
    ax.plot(f_p, db_p, color="tab:green", alpha=0.85, lw=0.9, label="Processed (после фильтра)")
    ax.plot(f_c, db_c, color="tab:blue", alpha=0.85, lw=0.9, label="Clean (эталон)")
    ax.set_xlim(0, scales.f_max_hz)
    ax.set_ylim(*scales.fft_db_lim)
    ax.set_title(f"Спектр FFT до и после фильтрации: {sp_id}, шум «{NOISE_TYPE}» ({snr_nominal_db} дБ)")
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("Уровень, дБ отн. полной шкалы")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_delta_snr(exp: pd.DataFrame, out_path: Path) -> Path:
    """Столбчатая диаграмма ΔSNR (дБ) по 12 записям, сгруппированным по уровням."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    after_df = exp[exp["stage"] == "after"].copy()
    if after_df.empty:
        raise ValueError("Нет данных stage == 'after'")

    sp_ids = sorted(after_df["sp_id"].unique())
    levels = sorted(after_df["snr_nominal_db"].unique())
    x = np.arange(len(levels))
    n_bars = len(sp_ids)
    width = 0.8 / max(n_bars, 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, sp_id in enumerate(sp_ids):
        sub = after_df[after_df["sp_id"] == sp_id].set_index("snr_nominal_db")
        vals = [float(sub.loc[lvl, "delta_snr"]) if lvl in sub.index else 0.0 for lvl in levels]
        offset = (idx - (n_bars - 1) / 2.0) * width
        bars = ax.bar(x + offset, vals, width=width * 0.92, label=sp_id)
        for bar, val in zip(bars, vals):
            va = "bottom" if val >= 0 else "top"
            y_pos = val + (0.08 if val >= 0 else -0.08)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos,
                f"{val:+.2f}",
                ha="center",
                va=va,
                fontsize=8,
            )

    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lvl} дБ" for lvl in levels])
    ax.set_xlabel("Номинальный входной SNR")
    ax.set_ylabel("ΔSNR, дБ")
    ax.set_title("Изменение отношения сигнал/шум (ΔSNR) по уровням шума и фразам")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Фраза (sp_id)")

    min_val = after_df["delta_snr"].min()
    max_val = after_df["delta_snr"].max()
    padding = max(0.01, (max_val - min_val) * 0.2) 
    ax.set_ylim(min_val - padding, max_val + padding)

    fig.savefig(out_path, dpi=DPI)
    
    plt.close(fig)
    return out_path


def make_listening_template(records: pd.DataFrame, out_csv: Path) -> Path:
    """Создаёт пустой CSV-шаблон для субъективного прослушивания командой."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, rec in records.iterrows():
        rows.append({
            "sp_id": str(rec["sp_id"]).lower(),
            "snr": int(rec["snr_nominal_db"]),
            "остаточный шум 1–5": "",
            "речь стала глуше 1–5": "",
            "артефакты 1–5": "",
            "комментарий": "",
        })
    df = pd.DataFrame(rows).sort_values(by=["snr", "sp_id"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out_csv


def run_stage5(
    dataset_dir: Path = DATASET_DIR,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Path]:
    """Точка входа Этапа 5. При отсутствии 12 WAV из Задачи 4 использует заглушку 0.9 * noisy."""
    ddr = Path(dataset_dir)
    rdr = Path(results_dir)
    tables_dir = rdr / "tables"
    figs_dir = rdr / "figures"
    audio_dir = rdr / "audio_processed"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    proc_csv = tables_dir / "processed_files.csv"
    temp_dir_obj = None

    if proc_csv.is_file():
        records = pd.read_csv(proc_csv)
        used_audio_dir = audio_dir
    else:
        logger.warning("Файл %s не найден. Запускается заглушка: processed = 0.9 * noisy.", proc_csv)
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="stage5_stub_")
        used_audio_dir = Path(temp_dir_obj.name)
        stub_rows = []
        for sp_id in SELECTED_IDS:
            for snr in SNR_LEVELS:
                pair = load_pair(sp_id, snr, dataset_dir=ddr)
                stub_sig = 0.9 * pair.noisy
                out_wav = used_audio_dir / f"{sp_id}_{NOISE_TYPE}_{snr}dB_filtered.wav"
                save_wav(out_wav, stub_sig, pair.fs, dataset_dir=ddr)
                stub_rows.append({
                    "sp_id": sp_id,
                    "snr_nominal_db": snr,
                    "out_path": str(out_wav),
                })
        records = pd.DataFrame(stub_rows)

    exp_df = build_experiment_table(records, FILTER, dataset_dir=ddr)
    exp_path = tables_dir / "experiments.csv"
    exp_df.to_csv(exp_path, index=False, encoding="utf-8-sig")

    summary = summarize_effect(exp_df)
    print(
        f"[Этап 5] Средний ΔSNR: {summary['overall_mean_delta_snr']:+.2f} дБ | "
        f"улучшено: {summary['n_positive']}, ухудшено: {summary['n_negative']}"
    )
    print(f"  По уровням: {summary['mean_delta_snr_by_level']}")
    print(f"  Лучшая: {summary['best_record']} | Худшая: {summary['worst_record']}")

    listen_path = make_listening_template(records, tables_dir / "listening_template.csv")

    outputs: dict[str, Path] = {
        "experiments": exp_path,
        "listening_template": listen_path,
        "delta_snr_plot": plot_delta_snr(exp_df, figs_dir / "stage5_delta_snr.png"),
    }

    for sp_id in SELECTED_IDS:
        all_sigs = {}
        fs_ref = None
        for snr in SNR_LEVELS:
            p = load_pair(sp_id, snr, dataset_dir=ddr)
            if fs_ref is None:
                fs_ref = p.fs
            all_sigs["clean"] = p.clean
            all_sigs[f"{snr}dB"] = p.noisy
        scales = compute_common_scales(all_sigs, fs_ref)

        for snr in SNR_LEVELS:
            fig_out = figs_dir / f"stage5_{sp_id}_{snr}dB_fft_before_after.png"
            outputs[f"fft{sp_id}_{snr}dB"] = plot_before_after_spectra(
                sp_id=sp_id,
                snr_nominal_db=snr,
                dataset_dir=ddr,
                out_path=fig_out,
                scales=scales,
                processed_dir=used_audio_dir,
            )

    if temp_dir_obj is not None:
        temp_dir_obj.cleanup()

    return outputs


if __name__ == "__main__":
    run_stage5(DATASET_DIR, RESULTS_DIR)