# Лабораторная работа №3 - Вариант 06 «Аэропорт»

**Цель:** Исследование и восстановление речевых записей с акустическими помехами.

## 1. Данные

- Всего файлов в датасете: 150 (30 чистых и 120 зашумленных).
- Частота дискретизации: 8000 Гц.
- Длительность записей: 2.116 – 3.508 с.
- Выбранные для детального анализа фразы: `sp03`, `sp14`, `sp26`.

## 2. Методика

### Формулы
- **RMS:** `sqrt( (1/N) * Σ(x_i^2) )`
- **SNR:** `10 * log10( Σ(clean_i^2) / Σ(clean_i - test_i)^2 )`
- **ΔSNR:** `SNR_после - SNR_до`

### Параметры выбранного фильтра
- Тип: highpass (фильтр верхних частот)
- Семейство: butter (Баттерворта)
- Порядок: 4
- Частота среза: 100.0 Гц
- Нулевая фаза (zero_phase): True (используется `sosfiltfilt`, чтобы не было сдвига сигнала).

## 3. Результаты

### Таблица экспериментов

| sp_id | noise_type | snr_nominal_db | stage | fs_hz | n_samples | filter_type | filter_family | filter_order | filter_cutoff_hz | filter_zero_phase | snr_db | delta_snr | rms | peak | n_clipped | ls_gain | wav_path | comment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sp03 | airport | 0 | before | 8000 | 17967 | — | — | — | — | — | -0.55 | — | 0.058 | 0.511 | 0 | 1.014 | — | исходный зашумлённый сигнал |
| sp03 | airport | 0 | after | 8000 | 17967 | highpass | butter | 4.0 | 100.0 | True | -0.56 | -0.01 | 0.058 | 0.511 | 0 | 1.014 | results\audio_processed\sp03_airport_0dB_filtered.wav | ухудшил. Шум широкополосный, фильтр убрал часть речи |
| sp03 | airport | 5 | before | 8000 | 17967 | — | — | — | — | — | 4.45 | — | 0.046 | 0.468 | 0 | 1.005 | — | исходный зашумлённый сигнал |
| sp03 | airport | 5 | after | 8000 | 17967 | highpass | butter | 4.0 | 100.0 | True | 4.45 | -0.00 | 0.046 | 0.468 | 0 | 1.005 | results\audio_processed\sp03_airport_5dB_filtered.wav | без изменений, шум в полосе речи |
| sp03 | airport | 10 | before | 8000 | 17967 | — | — | — | — | — | 9.45 | — | 0.042 | 0.454 | 0 | 1.001 | — | исходный зашумлённый сигнал |
| sp03 | airport | 10 | after | 8000 | 17967 | highpass | butter | 4.0 | 100.0 | True | 9.45 | -0.00 | 0.042 | 0.454 | 0 | 1.001 | results\audio_processed\sp03_airport_10dB_filtered.wav | без изменений, ФВЧ неэффективен |
| sp03 | airport | 15 | before | 8000 | 17967 | — | — | — | — | — | 14.45 | — | 0.040 | 0.443 | 0 | 1.002 | — | исходный зашумлённый сигнал |
| sp03 | airport | 15 | after | 8000 | 17967 | highpass | butter | 4.0 | 100.0 | True | 14.45 | 0.00 | 0.040 | 0.442 | 0 | 1.002 | results\audio_processed\sp03_airport_15dB_filtered.wav | слабое улучшение (убрали низкочастотный гул) |
| sp14 | airport | 0 | before | 8000 | 23466 | — | — | — | — | — | -0.39 | — | 0.069 | 0.638 | 0 | 1.033 | — | исходный зашумлённый сигнал |
| sp14 | airport | 0 | after | 8000 | 23466 | highpass | butter | 4.0 | 100.0 | True | -0.39 | 0.00 | 0.069 | 0.639 | 0 | 1.033 | results\audio_processed\sp14_airport_0dB_filtered.wav | слабое улучшение, гул убран |
| sp14 | airport | 5 | before | 8000 | 23466 | — | — | — | — | — | 4.61 | — | 0.055 | 0.649 | 0 | 1.006 | — | исходный зашумлённый сигнал |
| sp14 | airport | 5 | after | 8000 | 23466 | highpass | butter | 4.0 | 100.0 | True | 4.61 | 0.00 | 0.055 | 0.649 | 0 | 1.006 | results\audio_processed\sp14_airport_5dB_filtered.wav | без изменений |
| sp14 | airport | 10 | before | 8000 | 23466 | — | — | — | — | — | 9.61 | — | 0.049 | 0.614 | 0 | 0.998 | — | исходный зашумлённый сигнал |
| sp14 | airport | 10 | after | 8000 | 23466 | highpass | butter | 4.0 | 100.0 | True | 9.61 | 0.00 | 0.049 | 0.614 | 0 | 0.998 | results\audio_processed\sp14_airport_10dB_filtered.wav | без изменений |
| sp14 | airport | 15 | before | 8000 | 23466 | — | — | — | — | — | 14.61 | — | 0.048 | 0.622 | 0 | 1.002 | — | исходный зашумлённый сигнал |
| sp14 | airport | 15 | after | 8000 | 23466 | highpass | butter | 4.0 | 100.0 | True | 14.61 | -0.00 | 0.048 | 0.622 | 0 | 1.002 | results\audio_processed\sp14_airport_15dB_filtered.wav | ухудшил, отрезал полезные низкие частоты речи |
| sp26 | airport | 0 | before | 8000 | 19934 | — | — | — | — | — | -0.49 | — | 0.062 | 0.454 | 0 | 0.999 | — | исходный зашумлённый сигнал |
| sp26 | airport | 0 | after | 8000 | 19934 | highpass | butter | 4.0 | 100.0 | True | -0.48 | 0.00 | 0.062 | 0.454 | 0 | 0.999 | results\audio_processed\sp26_airport_0dB_filtered.wav | слабое улучшение |
| sp26 | airport | 5 | before | 8000 | 19934 | — | — | — | — | — | 4.51 | — | 0.050 | 0.474 | 0 | 1.014 | — | исходный зашумлённый сигнал |
| sp26 | airport | 5 | after | 8000 | 19934 | highpass | butter | 4.0 | 100.0 | True | 4.51 | 0.00 | 0.050 | 0.474 | 0 | 1.014 | results\audio_processed\sp26_airport_5dB_filtered.wav | слабое улучшение |
| sp26 | airport | 10 | before | 8000 | 19934 | — | — | — | — | — | 9.51 | — | 0.045 | 0.448 | 0 | 1.002 | — | исходный зашумлённый сигнал |
| sp26 | airport | 10 | after | 8000 | 19934 | highpass | butter | 4.0 | 100.0 | True | 9.51 | -0.00 | 0.045 | 0.448 | 0 | 1.002 | results\audio_processed\sp26_airport_10dB_filtered.wav | ухудшил |
| sp26 | airport | 15 | before | 8000 | 19934 | — | — | — | — | — | 14.51 | — | 0.043 | 0.455 | 0 | 1.001 | — | исходный зашумлённый сигнал |
| sp26 | airport | 15 | after | 8000 | 19934 | highpass | butter | 4.0 | 100.0 | True | 14.51 | -0.01 | 0.043 | 0.455 | 0 | 1.001 | results\audio_processed\sp26_airport_15dB_filtered.wav | ухудшил (ΔSNR -0.01), шум широкополосный |

### Рисунки

![stage2_sp03_fft](figures/stage2_sp03_fft.png)
![stage2_sp03_spectrogram](figures/stage2_sp03_spectrogram.png)
![stage2_sp03_waveform](figures/stage2_sp03_waveform.png)
![stage2_sp14_fft](figures/stage2_sp14_fft.png)
![stage2_sp14_spectrogram](figures/stage2_sp14_spectrogram.png)
![stage2_sp14_waveform](figures/stage2_sp14_waveform.png)
![stage2_sp26_fft](figures/stage2_sp26_fft.png)
![stage2_sp26_spectrogram](figures/stage2_sp26_spectrogram.png)
![stage2_sp26_waveform](figures/stage2_sp26_waveform.png)

![stage3_band_fractions](figures/stage3_band_fractions.png)
![stage3_measured_vs_nominal](figures/stage3_measured_vs_nominal.png)
![stage3_sp03_noise_vs_speech](figures/stage3_sp03_noise_vs_speech.png)
![stage3_sp14_noise_vs_speech](figures/stage3_sp14_noise_vs_speech.png)
![stage3_sp26_noise_vs_speech](figures/stage3_sp26_noise_vs_speech.png)

![stage4_filter_response](figures/stage4_filter_response.png)
![stage4_sp03_filter_vs_spectra](figures/stage4_sp03_filter_vs_spectra.png)
![stage4_sp14_filter_vs_spectra](figures/stage4_sp14_filter_vs_spectra.png)
![stage4_sp26_filter_vs_spectra](figures/stage4_sp26_filter_vs_spectra.png)

![stage5_delta_snr](figures/stage5_delta_snr.png)
![stage5_sp03_0dB_fft_before_after](figures/stage5_sp03_0dB_fft_before_after.png)
![stage5_sp03_10dB_fft_before_after](figures/stage5_sp03_10dB_fft_before_after.png)
![stage5_sp03_15dB_fft_before_after](figures/stage5_sp03_15dB_fft_before_after.png)
![stage5_sp03_5dB_fft_before_after](figures/stage5_sp03_5dB_fft_before_after.png)
![stage5_sp14_0dB_fft_before_after](figures/stage5_sp14_0dB_fft_before_after.png)
![stage5_sp14_10dB_fft_before_after](figures/stage5_sp14_10dB_fft_before_after.png)
![stage5_sp14_15dB_fft_before_after](figures/stage5_sp14_15dB_fft_before_after.png)
![stage5_sp14_5dB_fft_before_after](figures/stage5_sp14_5dB_fft_before_after.png)
![stage5_sp26_0dB_fft_before_after](figures/stage5_sp26_0dB_fft_before_after.png)
![stage5_sp26_10dB_fft_before_after](figures/stage5_sp26_10dB_fft_before_after.png)
![stage5_sp26_15dB_fft_before_after](figures/stage5_sp26_15dB_fft_before_after.png)
![stage5_sp26_5dB_fft_before_after](figures/stage5_sp26_5dB_fft_before_after.png)

## Вывод

В ходе работы было проанализировано 150 записей. Уровень шума в датасете ожидаемо падает при переходе от папки 0 дБ (средний измеренный SNR = -0.55 дБ) к папке 15 дБ (средний измеренный SNR = 14.45 дБ). Разница между нашим измеренным SNR и номинальным (по названию папки) составляет около -0.55 дБ. Это объясняется тем, что мы считаем SNR по всей записи, включая паузы, а в оригинальном корпусе NOIZEUS уровень задавался только по активной речи.

Для фильтрации был выбран фильтр верхних частот (ФВЧ) Баттерворта 4-го порядка с частотой среза 100 Гц. Выбор обоснован тем, что анализ шума (этап 3) показал наличие сильного низкочастотного гула в области до 200 Гц. 

Применение фильтра к 12 записям показало среднее изменение SNR (ΔSNR) около 0.00 дБ: для 6 записей SNR незначительно улучшился, для 6 — незначительно ухудшился. Лучший результат показала запись sp03 на 15 дБ (+0.00 дБ), худший — sp03 на 0 дБ (-0.01 дБ). Это ожидаемый результат: так как шум аэропорта является широкополосным и пересекается с речью, ФВЧ отсекает только низкочастотный гул, но не может убрать основной шум без вреда для самой речи.

## Ограничения

1. Подбор параметров фильтра опирался на наличие чистого эталона (clean), что в реальной жизни невозможно (проблема «оракула»).
2. Метрика SNR не равна разборчивости речи. Ухо может воспринимать звук лучше или хуже, чем показывают цифры.
3. Оценка шума как `noisy - clean` может содержать в себе искажения самой речи, а не только чистый шум.