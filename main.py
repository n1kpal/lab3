"""Точка входа, читает аргументы и запускает конвеер"""
import argparse
import sys
import time
from pathlib import Path

# Добавляем src в sys.path
src = Path(__file__).resolve().parent / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from lab3 import config
from lab3.inventory import run_stage1
from lab3.plots import run_stage2
from lab3.noise_analysis import run_stage3
from lab3.filtering import run_stage4
from lab3.evaluation import run_stage5
from lab3.report import run_stage6


# Пайплайн
def run_pipeline(dataset_dir, results_dir, stages):
    runners = {1: run_stage1, 2: run_stage2, 3: run_stage3, 4: run_stage4, 5: run_stage5, 6: run_stage6}

    results_dir.mkdir(parents=True, exist_ok=True)
    res = {}

    for n in stages:
        print(f"\nЭтап {n}")
        t0 = time.perf_counter()
        try:
            out = runners[n](dataset_dir, results_dir)
        except NotImplementedError as e:
            sys.exit(f"Этап {n} ещё не реализован: {e}")
        except Exception as e:
            sys.exit(f"Этап {n} упал: {type(e).__name__}: {e}")
        print(f"Этап {n} готов за {time.perf_counter() - t0:.1f} с; артефактов: {len(out)}")
        res[n] = out
    return res


if __name__ == "__main__":
    # Читаем аргументы
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=config.DATASET_DIR, help="папка с данными (по умолчанию: dataset)")
    p.add_argument("--results", type=Path, default=config.RESULTS_DIR, help="папка для результатов (по умолчанию: results)")
    p.add_argument("--stages", type=str, default="1,2,3,4,5,6", help="этапы через запятую, например: 1,2,3 (по умолчанию все этапы)")
    p.add_argument("--selected", type=str, default=None, help="три sp_id через запятую, например: sp05,sp17,sp23")
    args = p.parse_args()

    try:
        stages = tuple(map(int, args.stages.split(",")))
    except ValueError:
        sys.exit(f"Ошибка в --stages={args.stages}: ожидались числа через запятую")
    for i in stages:
        if i not in (1, 2, 3, 4, 5, 6):
            sys.exit(f"Неизвестные этапы: {stages}. Допустимо: 1..6")

    if args.selected:
        ids = tuple(args.selected.split(","))
        if len(ids) != 3:
            sys.exit(f"--selected требует 3 sp_id, получено {len(ids)}")
        config.SELECTED_IDS = ids


    # Проверяем датасет
    clean_dir = args.dataset / "clean"
    noisy_dir = args.dataset / "noisy" / config.NOISE_TYPE

    if clean_dir.is_dir():
        n_clean = len(list(clean_dir.glob("sp*.wav")))
    else:
        n_clean = 0

    n_noisy = 0
    for snr in config.SNR_LEVELS:
        cur = noisy_dir / f"{snr}dB"
        if cur.is_dir():
            n_noisy += len(list(cur.glob("sp*.wav")))

    if n_clean != 30 or n_noisy != 120:
        print(f"Датасет неполон: clean={n_clean}/30, noisy={n_noisy}/120", file=sys.stderr)
        print(f"Запустите: py tools/download_dataset.py --out {args.dataset}", file=sys.stderr)
        sys.exit(2)

    run_pipeline(args.dataset, args.results, stages)
    print("\nГотово.")