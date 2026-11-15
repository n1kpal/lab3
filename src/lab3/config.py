'''Единственный источник констант'''
from pathlib import Path

NOISE_TYPE = "airport"
SNR_LEVELS = (0, 5, 10, 15)
DATASET_DIR = Path("dataset")
RESULTS_DIR = Path("results")
SELECTED_IDS = ("sp03", "sp14", "sp26")
BANDS_HZ = ((0, 100), (100, 300), (300, 1000), (1000, 2000), (2000, 3400), (3400, 4000))
CLIP_THRESHOLD = 0.999

'''
Заполняет задача 4 после подбора, до тех пор — временные значения (ФВЧ 100 Гц, порядок 4).
Отчёт и оценка берут параметры только отсюда.
'''
FILTER = dict(kind="highpass", family="butter", order=4, cutoff_hz=100.0, zero_phase=True)