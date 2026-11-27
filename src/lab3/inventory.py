import logging
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

NOISE_TYPE = "airport"
SNR_LEVELS = (0, 5, 10, 15)
DATASET_DIR = Path("dataset")
RESULTS_DIR = Path("results")
SELECTED_IDS = ("sp03", "sp14", "sp26")
BANDS_HZ = ((0, 100), (100, 300), (300, 1000), (1000, 2000), (2000, 3400), (3400, 4000))
CLIP_THRESHOLD = 0.999
FILTER = {
    "kind": "highpass",
    "family": "butter",
    "order": 4,
    "cutoff_hz": 100.0,
    "zero_phase": True,
}

EXPECTED_IDS = []
for i in range(1, 31):
    EXPECTED_IDS.append(f"sp{i:02d}")
EXPECTED_IDS = tuple(EXPECTED_IDS)

INVENTORY_COLUMNS = [
    "rel_path", "kind", "sp_id", "noise_type", "snr_nominal_db",
    "duration_s", "fs_hz", "n_channels", "bit_depth", "n_samples",
    "min_amp", "max_amp", "peak_abs", "peak_dbfs", "rms", "rms_dbfs", "has_clean_pair",
]
PAIRS_COLUMNS = ["sp_id", "snr_nominal_db", "clean_path", "noisy_path", "ok", "problem"]

EPS = 1e-12
SP_PATTERN = re.compile(r"(sp\d{2})", re.IGNORECASE)
SNR_PATTERN = re.compile(r"^([+-]?\d+)\s*dB$", re.IGNORECASE)
BIT_DEPTHS = {
    "PCM_S8": 8,
    "PCM_U8": 8,
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT": 32,
    "DOUBLE": 64,
}

logger = logging.getLogger(__name__)
Signal = np.ndarray


class AlignmentError(ValueError):
    """Ошибка несовпадения длин сигналов в паре."""


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


# --- Метрики ---

def _check_signal(x: Signal, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} должен быть непустым одномерным массивом")
    return arr


def rms(x: Signal) -> float:
    arr = _check_signal(x, "x")
    return float(np.sqrt(np.mean(arr * arr)))


def snr_db(clean: Signal, test: Signal) -> float:
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


def delta_snr_db(snr_before: float, snr_after: float) -> float:
    return float(snr_after - snr_before)


def check_alignment(clean: Signal, test: Signal, max_lag: int = 400) -> AlignmentReport:
    if max_lag < 0:
        raise ValueError("max_lag должен быть >= 0")
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
    if denom > 0.0:
        corr = float(valid_xcorr[best_index] / denom)
    else:
        corr = 0.0

    min_len = min(c.size, t.size)
    c_part = c[:min_len]
    t_part = t[:min_len]
    c_energy = float(np.sum(c_part * c_part))
    if c_energy > 0.0:
        ls_gain = float(np.sum(c_part * t_part) / c_energy)
    else:
        ls_gain = 0.0

    return AlignmentReport(c.size == t.size, lag_samples, ls_gain, corr)

def load_wav(path: Path) -> tuple[Signal, int]:
    data, fs = sf.read(Path(path), dtype="float64", always_2d=False)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim > 1:
        if arr.shape[1] > 1:
            logger.warning("В файле %s %d каналов — взят 1-й канал.", path, arr.shape[1])
        arr = arr[:, 0]
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    return arr, int(fs)


def parse_wav_path(path: Path, dataset_dir: Path) -> dict:
    wav_path = Path(path)
    root = Path(dataset_dir)
    try:
        rel = wav_path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Файл {wav_path} находится вне папки {root}") from exc

    parts = rel.parts
    if len(parts) < 2:
        raise ValueError(f"Некорректный путь: {rel}")

    sp_match = SP_PATTERN.search(wav_path.stem)
    if sp_match is None:
        raise ValueError(f"Не найден sp_id (spNN) в имени файла: {wav_path.name}")
    sp_id = sp_match.group(1).lower()

    top_folder = parts[0].lower()
    if top_folder == "clean" and len(parts) == 2:
        return {
            "kind": "clean",
            "sp_id": sp_id,
            "noise_type": None,
            "snr_nominal_db": None,
        }

    if top_folder == "noisy" and len(parts) >= 4:
        snr_match = SNR_PATTERN.fullmatch(parts[2].strip())
        if snr_match is not None:
            return {
                "kind": "noisy",
                "sp_id": sp_id,
                "noise_type": parts[1],
                "snr_nominal_db": int(snr_match.group(1)),
            }

    raise ValueError(f"Неожиданная структура пути: {rel}")


def _find_one_wav(folder: Path, dataset_dir: Path, kind: str, sp_id: str, snr_db_val: int) -> Path:
    if not folder.exists():
        raise FileNotFoundError(f"Папка не найдена: {folder}")

    hits = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or p.suffix.lower() != ".wav":
            continue
        try:
            meta = parse_wav_path(p, dataset_dir)
        except ValueError:
            continue
        if meta["kind"] == kind and meta["sp_id"] == sp_id:
            if kind == "clean" or meta["snr_nominal_db"] == snr_db_val:
                hits.append(p)

    if len(hits) == 0:
        raise FileNotFoundError(f"Не найден {kind}-файл для {sp_id} ({snr_db_val} dB)")
    if len(hits) > 1:
        raise ValueError(f"Найдено несколько {kind}-файлов для {sp_id} ({snr_db_val} dB)")
    return hits[0]


def load_pair(
    sp_id: str,
    snr_nominal_db: int,
    dataset_dir: Path = DATASET_DIR,
    strict: bool = True,
) -> Pair:
    root = Path(dataset_dir)
    target_id = sp_id.lower()

    clean_path = _find_one_wav(root / "clean", root, "clean", target_id, snr_nominal_db)
    noisy_dir = root / "noisy" / NOISE_TYPE / f"{snr_nominal_db}dB"
    if not noisy_dir.exists():
        noisy_dir = root / "noisy"
    noisy_path = _find_one_wav(noisy_dir, root, "noisy", target_id, snr_nominal_db)

    clean_sig, fs_c = load_wav(clean_path)
    noisy_sig, fs_n = load_wav(noisy_path)

    if fs_c != fs_n:
        raise ValueError(f"Разный fs для {target_id} ({snr_nominal_db} dB): {fs_c} != {fs_n}")

    if len(clean_sig) != len(noisy_sig):
        if strict:
            raise AlignmentError(
                f"Разная длина для {target_id} ({snr_nominal_db} dB): {len(clean_sig)} != {len(noisy_sig)}"
            )
        min_len = min(len(clean_sig), len(noisy_sig))
        logger.warning("Обрезка пары %s (%d dB) до %d отсчётов.", target_id, snr_nominal_db, min_len)
        clean_sig = clean_sig[:min_len]
        noisy_sig = noisy_sig[:min_len]

    return Pair(target_id, int(snr_nominal_db), clean_sig, noisy_sig, fs_c)


def save_wav(
    path: Path,
    x: Signal,
    fs: int,
    dataset_dir: Path = DATASET_DIR,
    subtype: str = "PCM_16",
) -> dict:
    out_path = Path(path)
    if out_path.resolve().is_relative_to(Path(dataset_dir).resolve()):
        raise ValueError(f"Запрещено писать внутрь dataset_dir: {out_path}")

    arr = _check_signal(x, "x")
    abs_arr = np.abs(arr)
    n_clipped = int(np.sum(abs_arr >= CLIP_THRESHOLD))
    peak = float(np.max(abs_arr))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, arr, int(fs), subtype=subtype)
    return {"n_clipped": n_clipped, "peak": peak}


def _get_bit_depth(subtype: str) -> int | None:
    upper = subtype.upper()
    if upper in BIT_DEPTHS:
        return BIT_DEPTHS[upper]
    m = re.search(r"(\d+)", upper)
    if m is not None:
        return int(m.group(1))
    return None


def scan_dataset(dataset_dir: Path) -> pd.DataFrame:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Датасет не найден: {root}")

    wav_files = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() == ".wav":
            wav_files.append(p)

    root_res = root.resolve()
    rows = []
    for wav in wav_files:
        meta = parse_wav_path(wav, root)
        info = sf.info(wav)
        sig, fs_hz = load_wav(wav)

        n_samples = int(len(sig))
        if fs_hz > 0:
            duration_s = float(n_samples / fs_hz)
        else:
            duration_s = 0.0

        peak_abs = float(np.max(np.abs(sig)))
        rms_val = rms(sig)

        noise_type_str = meta["noise_type"]
        if noise_type_str is None:
            noise_type_str = ""

        rows.append({
            "rel_path": wav.resolve().relative_to(root_res).as_posix(),
            "kind": meta["kind"],
            "sp_id": meta["sp_id"],
            "noise_type": noise_type_str,
            "snr_nominal_db": meta["snr_nominal_db"],
            "duration_s": duration_s,
            "fs_hz": int(fs_hz),
            "n_channels": int(info.channels),
            "bit_depth": _get_bit_depth(info.subtype),
            "n_samples": n_samples,
            "min_amp": float(np.min(sig)),
            "max_amp": float(np.max(sig)),
            "peak_abs": peak_abs,
            "peak_dbfs": float(20.0 * np.log10(peak_abs + EPS)),
            "rms": rms_val,
            "rms_dbfs": float(20.0 * np.log10(rms_val + EPS)),
            "has_clean_pair": False,
        })

    if len(rows) == 0:
        df = pd.DataFrame(columns=INVENTORY_COLUMNS)
        df["snr_nominal_db"] = df["snr_nominal_db"].astype("Int64")
        df["has_clean_pair"] = df["has_clean_pair"].astype(bool)
        return df

    df = pd.DataFrame(rows)
    df["snr_nominal_db"] = pd.Series(df["snr_nominal_db"], dtype="Int64")
    df["bit_depth"] = pd.Series(df["bit_depth"], dtype="Int64")

    clean_ids = set(df.loc[df["kind"] == "clean", "sp_id"])
    noisy_ids = set(df.loc[df["kind"] == "noisy", "sp_id"])

    has_pair_list = []
    for _, row in df.iterrows():
        if row["kind"] == "noisy":
            has_pair_list.append(row["sp_id"] in clean_ids)
        else:
            has_pair_list.append(row["sp_id"] in noisy_ids)
    df["has_clean_pair"] = has_pair_list
    df["has_clean_pair"] = df["has_clean_pair"].astype(bool)

    df = df.sort_values(by=["kind", "snr_nominal_db", "sp_id", "rel_path"], na_position="first", kind="mergesort")
    df = df.reset_index(drop=True)
    return df[INVENTORY_COLUMNS]


def find_pairs(inv: pd.DataFrame) -> pd.DataFrame:
    """Сопоставить каждой noisy-записи из inv соответствующий clean-файл."""
    if inv.empty:
        return pd.DataFrame(columns=PAIRS_COLUMNS)

    clean_by_id = {}
    clean_df = inv[inv["kind"] == "clean"]
    for sid, grp in clean_df.groupby("sp_id", sort=False):
        clean_by_id[str(sid)] = grp

    rows = []
    noisy_df = inv[inv["kind"] == "noisy"]
    for _, nr in noisy_df.iterrows():
        sid = str(nr["sp_id"])
        snr = int(nr["snr_nominal_db"])
        n_path = str(nr["rel_path"])

        if sid not in clean_by_id:
            rows.append({
                "sp_id": sid,
                "snr_nominal_db": snr,
                "clean_path": "",
                "noisy_path": n_path,
                "ok": False,
                "problem": f"отсутствует clean-пара для {sid}",
            })
            continue

        c_grp = clean_by_id[sid]
        if len(c_grp) > 1:
            rows.append({
                "sp_id": sid,
                "snr_nominal_db": snr,
                "clean_path": str(c_grp.iloc[0]["rel_path"]),
                "noisy_path": n_path,
                "ok": False,
                "problem": f"дубликаты clean для {sid}",
            })
            continue

        cr = c_grp.iloc[0]
        probs = []
        if int(cr["fs_hz"]) != int(nr["fs_hz"]):
            probs.append(f"разный fs ({int(cr['fs_hz'])} != {int(nr['fs_hz'])})")
        if int(cr["n_samples"]) != int(nr["n_samples"]):
            probs.append(f"разная длина ({int(cr['n_samples'])} != {int(nr['n_samples'])})")

        rows.append({
            "sp_id": sid,
            "snr_nominal_db": snr,
            "clean_path": str(cr["rel_path"]),
            "noisy_path": n_path,
            "ok": len(probs) == 0,
            "problem": "; ".join(probs),
        })

    if len(rows) == 0:
        return pd.DataFrame(columns=PAIRS_COLUMNS)

    pairs_df = pd.DataFrame(rows)
    pairs_df = pairs_df.sort_values(by=["snr_nominal_db", "sp_id", "noisy_path"], kind="mergesort")
    pairs_df = pairs_df.reset_index(drop=True)
    return pairs_df[PAIRS_COLUMNS]


def _check_group(df_group: pd.DataFrame, label: str, expected: set[str], problems: list[str]) -> None:
    if len(df_group) != len(expected):
        problems.append(f"{label}: ожидалось {len(expected)} файлов, найдено {len(df_group)}")

    dups = sorted(df_group["sp_id"][df_group["sp_id"].duplicated()].astype(str).unique())
    if len(dups) > 0:
        problems.append(f"{label}: дубли sp_id: {', '.join(dups)}")

    actual = set(df_group["sp_id"].astype(str))
    missing = sorted(expected - actual)
    if len(missing) > 0:
        problems.append(f"{label}: отсутствуют sp_id: {', '.join(missing)}")

    extra = sorted(actual - expected)
    if len(extra) > 0:
        problems.append(f"{label}: лишние sp_id: {', '.join(extra)}")


def validate_inventory(inv: pd.DataFrame) -> list[str]:
    if inv.empty:
        return ["Инвентарь пуст: WAV-файлы не найдены."]

    problems = []
    expected = set(EXPECTED_IDS)

    clean_df = inv[inv["kind"] == "clean"]
    _check_group(clean_df, "clean", expected, problems)

    noisy_df = inv[inv["kind"] == "noisy"]
    bad_noise = sorted(set(noisy_df["noise_type"].astype(str)) - {NOISE_TYPE})
    if len(bad_noise) > 0:
        problems.append(f"Неожиданный тип шума: {', '.join(bad_noise)}")

    actual_snrs = set(noisy_df["snr_nominal_db"].dropna().astype(int))
    bad_snr = sorted(actual_snrs - set(SNR_LEVELS))
    if len(bad_snr) > 0:
        bad_snr_str = []
        for s in bad_snr:
            bad_snr_str.append(str(s))
        problems.append(f"Неожиданные уровни SNR: {', '.join(bad_snr_str)}")

    for snr in SNR_LEVELS:
        group_df = noisy_df[noisy_df["snr_nominal_db"] == snr]
        _check_group(group_df, f"noisy/{snr}dB", expected, problems)

    if inv["fs_hz"].nunique() > 1:
        dom_fs = int(inv["fs_hz"].mode().iloc[0])
        diffs = []
        for _, r in inv[inv["fs_hz"] != dom_fs].iterrows():
            diffs.append(f"{r['rel_path']} ({int(r['fs_hz'])} Гц)")
        problems.append(f"Разный fs (основной {dom_fs} Гц): {', '.join(diffs)}")

    return problems


def run_stage1(dataset_dir: Path, results_dir: Path) -> dict[str, Path]:
    tables_dir = Path(results_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    inv = scan_dataset(Path(dataset_dir))
    pairs = find_pairs(inv)
    problems = validate_inventory(inv)

    inv_path = tables_dir / "inventory.csv"
    pairs_path = tables_dir / "pairs.csv"
    inv.to_csv(inv_path, index=False, encoding="utf-8-sig")
    pairs.to_csv(pairs_path, index=False, encoding="utf-8-sig")

    if inv.empty:
        print(f"[Этап 1] Файлов: 0 | Проблем: {len(problems)}")
    else:
        fs_vals = sorted(inv["fs_hz"].dropna().astype(int).unique().tolist())
        bad_pairs = 0
        if not pairs.empty:
            bad_pairs = int((~pairs["ok"]).sum())
        dur_min = float(inv["duration_s"].min())
        dur_max = float(inv["duration_s"].max())
        print(
            f"[Этап 1] Файлов: {len(inv)}, пар: {len(pairs)} (проблемных: {bad_pairs}) | "
            f"fs: {fs_vals} Гц | длительность: {dur_min:.3f}..{dur_max:.3f} с | "
            f"проблем валидации: {len(problems)}"
        )

    for prob in problems:
        print(f"  - {prob}")

    return {"inventory": inv_path, "pairs": pairs_path}


if __name__ == "__main__":
    run_stage1(DATASET_DIR, RESULTS_DIR)