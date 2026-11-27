"""
анализ шумов в аудиозаписях и сравнение измеренного snr с номинальным

шум оценивается как разность noisy и clean. при этом в разности могут
оставаться искажения самой речи
=== Eрмолаев Eгор ===
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # рисование
import numpy as np  # массивы
import pandas as pd  # таблицы
from scipy import signal  # welch, csd

from lab3.config import BANDS_HZ, SELECTED_IDS, SNR_LEVELS
from lab3.inventory import find_pairs, scan_dataset, check_alignment, rms, snr_db, load_pair

NPERSEG_PSD = 512
EPS = 1e-12
SNR_ALL_COLUMNS = [
    "sp_id", "snr_nominal_db", "snr_measured_db", "diff_vs_nominal_db", "len_equal",
    "lag_samples", "ls_gain", "noise_rms", "noise_rms_dbfs", "noise_centroid_hz",
]


# =======================
# === вспомогательные ===
# =======================
def _band_label(lo: float, hi: float) -> str:
    return f"{lo:g}–{hi:g}"


def _band_powers(x: np.ndarray, fs: int, bands=BANDS_HZ) -> np.ndarray:
    """мощность сигнала в каждой полосе (линейная шкала, интеграл psd по полосе)"""
    f, pxx = signal.welch(np.asarray(x, dtype=float), fs=fs,
                          nperseg=min(NPERSEG_PSD, len(x)))
    df = f[1] - f[0]
    out = []
    last_hi = bands[-1][1]
    for lo, hi in bands:
        # верхняя граница последней полосы включается, остальные — нет (чтобы не считать дважды)
        upper = (f <= hi) if hi == last_hi else (f < hi)
        out.append(float(pxx[(f >= lo) & upper].sum() * df))
    return np.asarray(out)


# =============================
# === оценка шума и спектры ===
# =============================
def extract_noise(clean: np.ndarray, noisy: np.ndarray) -> np.ndarray:
    """возвращает разность noisy и clean после выравнивания сигналов"""
    clean = np.asarray(clean, dtype=float)
    noisy = np.asarray(noisy, dtype=float)
    if len(clean) != len(noisy):
        raise ValueError(f"разная длина: clean={len(clean)}, noisy={len(noisy)}")
    lag = check_alignment(clean, noisy).lag_samples
    if lag > 0:
        noisy_al, clean_al = noisy[lag:], clean[:-lag]
    elif lag < 0:
        noisy_al, clean_al = noisy[:lag], clean[-lag:]
    else:
        noisy_al, clean_al = noisy, clean
    if lag != 0:
        print(f"лаг {lag} отсчётов: шум выровнен обрезкой, длина стала короче")
    return noisy_al - clean_al


def welch_psd(x: np.ndarray, fs: int, nperseg: int = NPERSEG_PSD) -> tuple[np.ndarray, np.ndarray]:
    """спектральная плотность мощности в дб/гц"""
    x = np.asarray(x, dtype=float)
    f, pxx = signal.welch(x, fs=fs, nperseg=min(nperseg, len(x)))
    return f, 10.0 * np.log10(pxx + EPS)


def band_energy_fractions(x: np.ndarray, fs: int, bands=BANDS_HZ) -> pd.Series:
    """доля энергии в каждой полосе. сумма по полосам = 1 (энергия вне bands_hz не учитывается)"""
    p = _band_powers(x, fs, bands)
    total = p.sum()
    if total <= 0:
        raise ValueError("сигнал без энергии в заданных полосах")
    labels = [_band_label(lo, hi) for lo, hi in bands]
    return pd.Series(p / total, index=labels, name="доля энергии")


def local_snr_by_band(clean: np.ndarray, noise: np.ndarray, fs: int, bands=BANDS_HZ) -> pd.Series:
    """локальный snr в каждой полосе, дб. значения < 0 — речь и шум пересекаются в полосе"""
    ec = _band_powers(clean, fs, bands)
    en = _band_powers(noise, fs, bands)
    labels = [_band_label(lo, hi) for lo, hi in bands]
    return pd.Series(10.0 * np.log10((ec + EPS) / (en + EPS)), index=labels, name="local_snr_db")


def speech_noise_overlap(f_hz, speech_psd_db, noise_psd_db) -> dict:
    """где кривые psd речи и шума пересекаются и где шум >= речи"""
    f = np.asarray(f_hz, dtype=float)
    diff = np.asarray(noise_psd_db, dtype=float) - np.asarray(speech_psd_db, dtype=float)
    # точки пересечения: там, где знак разности меняется
    idx = np.where(np.diff(np.sign(diff)) != 0)[0]
    crossover = [float((f[i] + f[i + 1]) / 2) for i in idx]
    # интервалы, где шум >= речи (непрерывные отрезки)
    mask = diff >= 0
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    bands = [(float(f[a]), float(f[b])) for a, b in runs]
    return {"crossover_hz": crossover, "noise_dominant_bands": bands}


def estimate_speech_gain(clean: np.ndarray, noisy: np.ndarray, fs: int,
                         nperseg: int = NPERSEG_PSD) -> tuple[np.ndarray, np.ndarray]:
    """вычисляет отношение взаимного спектра сигналов к спектру clean в дб"""
    clean = np.asarray(clean, dtype=float)
    noisy = np.asarray(noisy, dtype=float)
    nps = min(nperseg, len(clean))
    f, pcc = signal.welch(clean, fs=fs, nperseg=nps)
    _, pcn = signal.csd(clean, noisy, fs=fs, nperseg=nps)
    gain_db = 10.0 * np.log10((np.abs(pcn) + EPS) / (pcc + EPS))
    return f, gain_db


# ===================================
# === измерение snr по всем парам ===
# ===================================
def measure_snr_all(pairs: pd.DataFrame, dataset_dir: Path) -> pd.DataFrame:
    """вычисляет измеренный snr для всех корректных пар записей"""
    dataset_dir = Path(dataset_dir)
    rows = []
    for _, r in pairs.iterrows():
        if not bool(r["ok"]):
            print(f"пропуск {r['sp_id']} {r['snr_nominal_db']} дб: {r['problem']}")
            continue
        sp_id, snr_nom = str(r["sp_id"]), int(r["snr_nominal_db"])
        pair = load_pair(sp_id, snr_nom, dataset_dir=dataset_dir, strict=True)
        measured = snr_db(pair.clean, pair.noisy)  # формула тз: clean как эталон
        rep = check_alignment(pair.clean, pair.noisy)  # только диагностика
        noise = pair.noisy - pair.clean  # оценка шума
        nr = rms(noise)
        f, pxx = signal.welch(noise, fs=pair.fs, nperseg=min(NPERSEG_PSD, len(noise)))
        total = float(np.sum(pxx))
        centroid = float(np.sum(f * pxx) / total) if total > 0 else float("nan")
        rows.append({
            "sp_id": sp_id,
            "snr_nominal_db": snr_nom,
            "snr_measured_db": float(measured),
            "diff_vs_nominal_db": float(measured - snr_nom),
            "len_equal": bool(rep.len_equal),
            "lag_samples": int(rep.lag_samples),
            "ls_gain": float(rep.ls_gain),
            "noise_rms": float(nr),
            "noise_rms_dbfs": float(20.0 * np.log10(nr + EPS)),
            "noise_centroid_hz": centroid,
        })
    return pd.DataFrame(rows, columns=SNR_ALL_COLUMNS)


def summarize_by_level(snr_all: pd.DataFrame) -> pd.DataFrame:
    """статистика по номиналам 0/5/10/15 дб: как меняется уровень помех между папками"""
    g = snr_all.groupby("snr_nominal_db")
    out = pd.DataFrame({
        "n": g.size(),
        "snr_mean_db": g["snr_measured_db"].mean(),
        "snr_std_db": g["snr_measured_db"].std(ddof=1),
        "snr_min_db": g["snr_measured_db"].min(),
        "snr_max_db": g["snr_measured_db"].max(),
        "noise_rms_mean": g["noise_rms"].mean(),
        "noise_rms_std": g["noise_rms"].std(ddof=1),
        "noise_rms_min": g["noise_rms"].min(),
        "noise_rms_max": g["noise_rms"].max(),
    })
    return out.reset_index()


def band_fractions_by_level(pairs: pd.DataFrame, dataset_dir: Path) -> pd.DataFrame:
    """средняя доля энергии шума по полосам для каждого уровня (строки — уровни)"""
    dataset_dir = Path(dataset_dir)
    labels = [_band_label(lo, hi) for lo, hi in BANDS_HZ]
    rows = {}
    ok = pairs["ok"].astype(bool)
    for snr in SNR_LEVELS:
        sub = pairs[ok & (pairs["snr_nominal_db"] == snr)]
        fr = []
        for _, r in sub.iterrows():
            pair = load_pair(str(r["sp_id"]), int(snr), dataset_dir=dataset_dir, strict=True)
            fr.append(band_energy_fractions(pair.noisy - pair.clean, pair.fs).values)
        rows[int(snr)] = np.mean(fr, axis=0)
    return pd.DataFrame.from_dict(rows, orient="index", columns=labels)


# ===============
# === графики ===
# ===============
def plot_noise_vs_speech(sp_id: str, dataset_dir: Path, out_path: Path) -> Path:
    """psd речи и psd шума для 0/5/10/15 дб на одном рисунке"""
    dataset_dir = Path(dataset_dir)
    first = load_pair(sp_id, SNR_LEVELS[0], dataset_dir=dataset_dir, strict=True)
    fs = first.fs
    fig, ax = plt.subplots(figsize=(9, 5))
    f, p_speech = welch_psd(first.clean, fs)
    ax.plot(f, p_speech, color="black", lw=2, label="речь (clean)")
    for snr in SNR_LEVELS:
        pair = load_pair(sp_id, snr, dataset_dir=dataset_dir, strict=True)
        f, p_noise = welch_psd(pair.noisy - pair.clean, fs)
        ax.plot(f, p_noise, alpha=0.8, label=f"шум (noisy − clean), {snr} дБ")
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("Спектральная плотность, дБ/Гц")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"{sp_id}: спектр речи и шума по уровням (оценка шума — допущение)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)


def plot_measured_vs_nominal(snr_all: pd.DataFrame, out_path: Path) -> Path:
    """измеренный snr против номинала: точки с разбросом по x и линия y = x"""
    rng = np.random.default_rng(0)
    x = snr_all["snr_nominal_db"].to_numpy(dtype=float)
    x = x + rng.uniform(-0.5, 0.5, size=len(x))
    y = snr_all["snr_measured_db"].to_numpy(dtype=float)
    lo = min(-1.0, float(np.nanmin(y)) - 1.0)
    hi = max(16.0, float(np.nanmax(y)) + 1.0)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=18, alpha=0.7, label="пары (120 шт.)")
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = x (идеальное совпадение)")
    ax.set_xlabel("Номинальный SNR (по папке), дБ")
    ax.set_ylabel("Измеренный SNR по формуле ТЗ, дБ")
    ax.set_xlim(-1, 16)
    ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Измеренный и номинальный SNR (расхождение показано, не подогнано)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)


def plot_band_fractions(fractions: pd.DataFrame, out_path: Path) -> Path:
    """доля энергии шума по полосам, столбики по уровням"""
    labels = list(fractions.columns)
    x = np.arange(len(labels))
    n = max(len(fractions.index), 1)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (lvl, row) in enumerate(fractions.iterrows()):
        ax.bar(x - 0.4 + width / 2 + i * width, row.values * 100, width,
               label=f"шум, {int(lvl)} дБ")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_xlabel("Полоса частот, Гц")
    ax.set_ylabel("Доля энергии шума, %")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Где сосредоточен шум: доля энергии по полосам BANDS_HZ", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)


# ====================
# === этап целиком ===
# ====================
def run_stage3(dataset_dir: Path, results_dir: Path) -> dict[str, Path]:
    dataset_dir, results_dir = Path(dataset_dir), Path(results_dir)
    tables = results_dir / "tables"
    figures = results_dir / "figures"
    for d in (tables, figures):
        d.mkdir(parents=True, exist_ok=True)

    # подготовка пар и расчёт snr
    inv = scan_dataset(dataset_dir)
    pairs = find_pairs(inv)
    n_ok = int(pairs["ok"].astype(bool).sum())
    if n_ok != 120:
        print(f"пригодных пар {n_ok} из 120 — проверьте pairs.csv (задача 1)")
    snr_all = measure_snr_all(pairs, dataset_dir)
    csv_path = tables / "noise_snr_all.csv"
    snr_all.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary = summarize_by_level(snr_all)
    fractions = band_fractions_by_level(pairs, dataset_dir)

    # сравнение спектров речи и шума при уровне 0 дб
    speech_list, noise_list = [], []
    for sp in SELECTED_IDS:
        pair0 = load_pair(sp, SNR_LEVELS[0], dataset_dir=dataset_dir, strict=True)
        f, ps = welch_psd(pair0.clean, pair0.fs)
        _, pn = welch_psd(pair0.noisy - pair0.clean, pair0.fs)
        speech_list.append(ps)
        noise_list.append(pn)
    speech_db = np.mean(speech_list, axis=0)  # усреднение в дб — упрощение, записать в отчёт
    noise_db = np.mean(noise_list, axis=0)
    overlap = speech_noise_overlap(f, speech_db, noise_db)

    # сохранение результатов анализа
    level_summary = {
        str(int(row["snr_nominal_db"])): {c: float(v) for c, v in row.items() if c != "snr_nominal_db"}
        for _, row in summary.iterrows()
    }
    findings = {
        "noise_dominant_bands": [[a, b] for a, b in overlap["noise_dominant_bands"]],
        "crossover_hz": overlap["crossover_hz"],
        "level_summary": level_summary,
        "mean_diff_vs_nominal_db": float(snr_all["diff_vs_nominal_db"].mean()),
        "band_fractions_by_level": {
            str(int(lvl)): {k: float(v) for k, v in row.items()}
            for lvl, row in fractions.iterrows()
        },
        "reference_level_db": int(SNR_LEVELS[0]),
        "selected_ids": list(SELECTED_IDS),
    }
    findings_path = tables / "findings.json"
    with open(findings_path, "w", encoding="utf-8") as fh:
        json.dump(findings, fh, ensure_ascii=False, indent=2)

    # построение графиков
    paths: dict[str, Path] = {"noise_snr_all": csv_path, "findings": findings_path}
    for sp in SELECTED_IDS:
        paths[f"noise_vs_speech_{sp}"] = plot_noise_vs_speech(
            sp, dataset_dir, figures / f"stage3_{sp}_noise_vs_speech.png")
    paths["measured_vs_nominal"] = plot_measured_vs_nominal(
        snr_all, figures / "stage3_measured_vs_nominal.png")
    paths["band_fractions"] = plot_band_fractions(
        fractions, figures / "stage3_band_fractions.png")

    print(f"этап 3: пар {len(snr_all)}, средняя разница измерено−номинал "
          f"{findings['mean_diff_vs_nominal_db']:.2f} дб, "
          f"шум сильнее речи на полосах: {findings['noise_dominant_bands']}")
    return paths