from pathlib import Path
import json
import pandas as pd
from lab3 import config
from lab3.evaluation import summarize_effect
"""
Path - для работы с путями
оыщт - для чтения
pandas  - для табличек csv
Импортируем из lab3 то что Федяйность сделал и кирилльность
"""
#Превращаем нашу датафреймовую табличку в markdown
def df_to_markdown(df: pd.DataFrame) -> str:
    #Так как я не могу посмотреть а Егор писал что могут быть пустые моменты то просто на всякий напишу вот это
    if len(df) == 0:
        return "_(таблица пуста)_" 
    #Сначала заголовки
    headrs = []
    #Циклом по всем колонкам проходим и записываем названия колонок
    for name in df.columns:
        headrs.append(str(name))
    #Склейка по названиям колонок. тут как раз нужен json
    headr_string = "| " + " | ".join(headrs) + " |"
    #Разделитель делаем. оказывается что только так может определить нашу табличку
    parts = []
    for i in df.columns:
        parts.append("---")
    string = "| " + " | ".join(parts) + " |"
    #Собираем строки с данными
    data_rows = []
    for index, row in df.iterrows():
        #список для ячеек одной строчки
        row_cells = []
        for name in df.columns:
            value = row[name]
            #Если пусто
            if pd.isna(value):
                #Ставлю прочерк
                row_cells.append("—")
            elif isinstance(value, float):
                #Если флоат то до двух знаков округляю
                row_cells.append(f"{value:.2f}")
            else:
                #Ну и все остальное в строку. почему бы и нет
                row_cells.append(str(value))
        #Как обычно склейка
        row_string = "| " + " | ".join(row_cells) + " |"
        data_rows.append(row_string)
    #А теперь все вместе
    all_lines = [headr_string, string] + data_rows
    return "\n".join(all_lines)
#Смотрит задачки ребят а также помещает в них пути к расположению файлов
def collect_artifacts(results_dir: Path) -> dict:
    results_dir = Path(results_dir)
    #Словарики для файлов
    tables = {} #Для csv
    figures = {}  #Для png
    audio = {}  #Для wav
    findings_data = None #для json. когда писал третья задачка была не готова так что если она все еще не сделана выведется "ничего". Всяко лучше того чтобы упала ошибкой
    #путь к папке с таблицами
    tables_dir = results_dir / "tables"
    if tables_dir.exists():
        #Все файлы с расширением .csv
        for file_path in tables_dir.glob("*.csv"):
            #Берем имена и пишем в словарь наш
            file_name = file_path.stem
            tables[file_name] = file_path
        #Отдельно json формируем
        json_path = tables_dir / "findings.json"
        if json_path.exists():
            #Открываем для чтения в нужной кодировке
            with open(json_path, "r", encoding="utf-8") as f:
                findings_data = json.load(f)
    #Точно также делаем с картинками и аудио файлами
    figures_dir = results_dir / "figures"
    if figures_dir.exists():
        for file_path in figures_dir.glob("*.png"):
            file_name = file_path.stem 
            figures[file_name] = file_path
    audio_dir = results_dir / "audio_processed"
    if audio_dir.exists():
        for file_path in audio_dir.glob("*.wav"):
            file_name = file_path.stem
            audio[file_name] = file_path
    #А теперь все в один гигасловарь
    result = {
        "tables": tables,
        "figures": figures,
        "audio": audio,
        "findings": findings_data
    }
    return result
#Пишет более понятный вывод
def conclusion_draft(findings: dict, effect: dict) -> str:
    text_parts = []
    text_parts.append("### Черновик вывода \n")
    #Шум
    text_parts.append("**1)Как меняется уровень шума:**")
    if findings is not None and "level_summary" in findings:
        level_dict = findings["level_summary"]
        #Проходимся по уровню шума
        for level_name in level_dict:
            #Конуретный уровень берем
            level_info = level_dict[level_name]
            if "snr_mean_db" in level_info:
                #берем и пишем средний уровень
                snr_val = level_info["snr_mean_db"]
                text_parts.append(f"- {level_name} дБ: средний измеренный SNR = {snr_val:.2f} дБ")
            else:
                text_parts.append(f"- {level_name} дБ: данных о среднем SNR нет")
    else:
        #Пока что затычка так как я не получил 3 задачу
        text_parts.append("- (данные из findings.json отсутствуют)")
    #Фильтр
    text_parts.append("\n**2)Почему выбран этот фильтр:**")
    #Достаем фильтр
    f = config.FILTER
    #Описание самого фильтра
    kind = f.get('kind', '?')
    family = f.get('family', '?')
    order = f.get('order', '?')
    cutoff = f.get('cutoff_hz', '?')
    zero_phase = f.get('zero_phase', '?')
    #А теперь мое любимое. Пишем гигастрокой
    text_parts.append(f"- Тип: {kind}, семейство: {family}, порядок: {order}, срез: {cutoff} Гц, zero_phase: {zero_phase}")
    if findings is not None and "noise_dominant_bands" in findings:
        #Если шум сильный то берем списко полос
        bands = findings["noise_dominant_bands"]
        bands_strings = []
        #Смотрим каждую полосу
        for i in bands:
            #В герцах
            bands_strings.append(f"{i[0]:g}–{i[1]:g} Гц")
            #Склейка
        text_parts.append(f"- Шум доминирует в полосах: {', '.join(bands_strings)}")
    #Справился ли фильтр?
    text_parts.append("\n**3)На каких записях фильтр помог а где нет:**")
    if effect is not None and len(effect) > 0:
        #Тут в общем киря говорил про это так что я просто напишу и выведу с умным лицом
        mean_delta = effect.get('overall_mean_delta_snr', 0.0)
        best_dict = effect.get('best_record', {})
        best_str = f"{best_dict.get('sp_id', '?')} (+{best_dict.get('delta_snr', 0.0):.2f} дБ)"
        worst_dict = effect.get('worst_record', {})
        worst_str = f"{worst_dict.get('sp_id', '?')} ({worst_dict.get('delta_snr', 0.0):.2f} дБ)"
        text_parts.append(f"- Средний ΔSNR: {mean_delta:.2f} дБ")
        text_parts.append(f"- Лучшая запись: {best_str}")
        text_parts.append(f"- Худшая запись: {worst_str}")
    else: #тоже затычка
        text_parts.append("- (данные из evaluation (задача 5) отсутствуют)")

    #Оковы(
    #Егор попросил написать ограничения. там по условию было
    text_parts.append("\n**4)Ограничения метода:**")
    text_parts.append("- Подбор параметров фильтра по эталону clean — «оракул» (в реальности чистого сигнала нет)")
    text_parts.append("- SNR не равен разборчивости речи")
    text_parts.append("- Оценка шума как noisy − clean может содержать искажения самой речи")
    return "\n".join(text_parts)
#Сам отчет
def render_report_md(ctx: dict) -> str:
    lines = []
    lines.append("# Лабораторная работа №3 - Вариант 06 «Аэропорт»\n")
    lines.append("**Цель:** Исследование и восстановление речевых записей с акустическими помехами\n")
    #Первый раздел
    lines.append("## 1)Данные\n")
    if "inventory" in ctx and len(ctx["inventory"]) > 0:
        #Ссылка
        inv = ctx["inventory"]
        total_files = len(inv)
        #Сборка частот
        unique_fs = []
        #Проходимся по всем частотам
        for fs_val in inv['fs_hz']:
            #Если нет то добавим
            if fs_val not in unique_fs:
                unique_fs.append(fs_val)
        min_dur = inv['duration_s'].min()
        max_dur = inv['duration_s'].max()
        lines.append(f"- Всего файлов: {total_files}")
        lines.append(f"- Частота дискретизации: {unique_fs} Гц")
        lines.append(f"- Длительность: {min_dur:.3f} – {max_dur:.3f} с")
    else:
        #Ладно раз начал так писать то везде уже допишу
        lines.append("- (нет данных инвентаризации)")
    #Склейка
    selected_str = ", ".join(config.SELECTED_IDS)
    lines.append(f"- Выбранные фразы: {selected_str}\n")
    #Второй раздел
    lines.append("## 2)Методика\n")
    lines.append("### Формулы")
    lines.append("- **RMS:** $\\sqrt{\\frac{1}{N}\\sum x_i^2}$")
    lines.append("- **SNR:** $10 \\lg \\frac{\\sum clean_i^2}{\\sum (clean_i - test_i)^2}$")
    #Ради этого я даже треугольник нашел
    lines.append("- **ΔSNR:** $SNR_{после} - SNR_{до}$")
    lines.append("\n### Параметры фильтра")
    #Достаем фильтр
    f = config.FILTER
    lines.append(f"- Тип: {f.get('kind')}")
    lines.append(f"- Порядок: {f.get('order')}")
    lines.append(f"- Частота среза: {f.get('cutoff_hz')} Гц")
    lines.append(f"- Нулевая фаза: {f.get('zero_phase')}\n")
    #Третий раздел
    lines.append("## 3)Результаты\n")
    if "experiments" in ctx and len(ctx["experiments"]) > 0:
        lines.append("### Таблица экспериментов (фрагмент)\n")
        small_table = ctx["experiments"].head(10)
        #В табличку markdown
        lines.append(df_to_markdown(small_table))
    lines.append("\n### Рисунки\n")
    if "figures" in ctx:
        #Список имен картинок
        figure_names = list(ctx["figures"].keys())
        #Сортировка по алфавиту
        figure_names.sort()
        for name in figure_names:
            lines.append(f"![{name}](figures/{name}.png)\n")
    #Четвертый раздел
    lines.append("## 4)Вывод\n")
    if "conclusion" in ctx:
        lines.append(ctx["conclusion"])
    #Пятый раздел
    lines.append("\n## 5)Ограничения метода\n")
    lines.append("- Подбор параметров фильтра по эталону clean — «оракул».")
    lines.append("- SNR не равен разборчивости речи.")
    lines.append("- Оценка шума как noisy − clean может содержать искажения речи.")
    return "\n".join(lines)
#Отчет в html. Тут я совсем перестал понимать поэтому обратился за помощью
def render_report_html(ctx: dict, results_dir: Path) -> str:
    results_dir = Path(results_dir)
    html_parts = []
    html_parts.append("<html><head><meta charset='utf-8'><title>ЛР3 Отчёт</title>")
    html_parts.append("<style>body{font-family:sans-serif;max-width:900px;margin:auto;padding:20px;}")
    html_parts.append("table{border-collapse:collapse;width:100%;margin:10px 0;}th,td{border:1px solid #ddd;padding:8px;text-align:left;}th{background-color:#f2f2f2;}")
    html_parts.append("audio{width:100%;margin:5px 0;}</style></head><body>")
    html_parts.append("<h1>Лабораторная работа №3</h1>")
    html_parts.append("<p><b>Вариант:</b> 06 «Аэропорт»</p>")
    if "experiments" in ctx and len(ctx["experiments"]) > 0:
        html_parts.append("<h2>Таблица экспериментов</h2>")
        html_table = ctx["experiments"].to_html(index=False, classes="table")
        html_parts.append(html_table)
    html_parts.append("<h2>Аудио</h2>")
    if "audio" in ctx and len(ctx["audio"]) > 0:
        #Список имен
        audio_names = list(ctx["audio"].keys())
        if len(audio_names) > 5:
            #Берем первые 5
            audio_names = audio_names[:5]
        for name in audio_names:
            path = ctx["audio"][name]
            rel_path = path.relative_to(results_dir)
            #Вставляем путь
            html_parts.append(f"<p><b>{name}</b><br><audio controls src='{rel_path.as_posix()}'></audio></p>")
    else:
        html_parts.append("<p>(аудиофайлы не найдены)</p>")
    html_parts.append("</body></html>")
    return "\n".join(html_parts)
#Проверка файлов
def check_deliverables(results_dir: Path) -> list[str]:
    results_dir = Path(results_dir)
    problems = []
    #Проверка inventory
    inv_path = results_dir / "tables" / "inventory.csv"
    if not inv_path.exists():
        #Если нет то пишем что отсутствует путь
        problems.append("Нет inventory.csv")
    else:
        #Если есть то читаем и пишем
        inv_df = pd.read_csv(inv_path, encoding="utf-8-sig")
        if len(inv_df) < 150:
            problems.append(f"inventory.csv: {len(inv_df)} строк, ожидалось >= 150")
    #Также делаем касаемо всего
    snr_path = results_dir / "tables" / "noise_snr_all.csv"
    if not snr_path.exists():
        problems.append("Нет noise_snr_all.csv")
    figures_dir = results_dir / "figures"
    count_figs = 0
    if figures_dir.exists():
        for f in figures_dir.glob("stage2_*.png"):
            count_figs += 1
    if count_figs < 9:
        problems.append(f"Рисунков stage2: {count_figs}, ожидалось >= 9")
    audio_dir = results_dir / "audio_processed"
    count_wav = 0
    if audio_dir.exists():
        for w in audio_dir.glob("*.wav"):
            count_wav += 1
    if count_wav < 12:
        problems.append(f"WAV в audio_processed: {count_wav}, ожидалось >= 12")
    exp_path = results_dir / "tables" / "experiments.csv"
    if not exp_path.exists():
        problems.append("Нет experiments.csv")
    else:
        exp_df = pd.read_csv(exp_path, encoding="utf-8-sig")
        if len(exp_df) < 24:
            problems.append(f"experiments.csv: {len(exp_df)} строк, ожидалось >= 24")
        #А тут ручная проверка на пустые snr
        if "snr_db" in exp_df.columns:
            has_empty = False
            #Проходимся по значениям
            for val in exp_df["snr_db"]:
                if pd.isna(val):
                    has_empty = True
                    break
            if has_empty:
                problems.append("experiments.csv: есть пустые snr_db")
    #Проверка небольшая
    report_md = results_dir / "report.md"
    if not report_md.exists():
        problems.append("Нет report.md")
    else:
        with open(report_md, "r", encoding="utf-8") as f:
            text = f.read()
        if "4) Вывод" not in text:
            problems.append("report.md: нет раздела 'Вывод'")
        if "5) Ограничения" not in text:
            problems.append("report.md: нет раздела 'Ограничения'")
    return problems
#Главная функция которая собирает все равнее написанное воедино
def run_stage6(dataset_dir: Path, results_dir: Path) -> dict[str, Path]:
    results_dir = Path(results_dir)
    all_files = collect_artifacts(results_dir)
    context = {}
    #записываем пути и читаем
    if "inventory" in all_files["tables"]:
        inv_path = all_files["tables"]["inventory"]
        context["inventory"] = pd.read_csv(inv_path, encoding="utf-8-sig")
    if "experiments" in all_files["tables"]:
        exp_path = all_files["tables"]["experiments"]
        context["experiments"] = pd.read_csv(exp_path, encoding="utf-8-sig")
    #делаем вывод
    findings = all_files["findings"]
    if "experiments" in all_files["tables"]:
        exp_df = context["experiments"]
        effect_data = summarize_effect(exp_df)
    else:
        effect_data = {}
    context["conclusion"] = conclusion_draft(findings, effect_data)
    context["figures"] = all_files["figures"]
    context["audio"] = all_files["audio"]
    #путь для markdown
    md_path = results_dir / "report.md"
    md_text = render_report_md(context)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    #Для html
    html_path = results_dir / "report.html"
    html_text = render_report_html(context, results_dir)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_text)
    #Проверка что все посмотрели
    issues = check_deliverables(results_dir)
    if len(issues) > 0:
        print(f"ВНИМАНИЕ: {len(issues)} проблем:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("Все проверки пройдены!")
    result_dict = {
        "report_md": md_path,
        "report_html": html_path,
    }
    return result_dict 