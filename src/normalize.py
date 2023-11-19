import pandas as pd
import sqlite3
from autocorrect import Speller
import translators as ts
import roman
import re
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy2


def normalize():
    # Подключение к базе данных
    db_path = '../db/data.db'
    conn = sqlite3.connect(db_path)

    # Загрузка конфига
    with open('../cfg/config.json', 'r', encoding="utf-8") as file:
        cfg = json.load(file)

    # Загрузка данных из каждой таблицы в отдельный DataFrame
    person_df = pd.read_sql("SELECT * FROM person", conn)
    education_df = pd.read_sql("SELECT * FROM education", conn)
    jobs_df = pd.read_sql("SELECT * FROM jobs", conn)

    def to_lower(text: str) -> str:
        return text.lower()

    def cut(text: str) -> str:
        symbols_for_cut = [",", "(", ";", ":"]
        for symbol in symbols_for_cut:
            idx = text.find(symbol)
            if idx != -1:
                text = text[:idx]
        return text

    jobs_df['job_name_norm'] = jobs_df['job_name'].apply(to_lower)
    jobs_df['job_name_norm'] = jobs_df['job_name_norm'].apply(cut)

    jobs_df['job_name_norm'] = jobs_df['job_name_norm'].str.replace(".", " ")
    jobs_df['job_name_norm'] = jobs_df['job_name_norm'].str.replace("ё", "е")

    # Использование регулярного выражения для замены двух или более пробелов на один пробел
    def replace_multiple_spaces(input_string):
        return re.sub(' +', ' ', input_string)

    jobs_df['job_name_norm'] = jobs_df['job_name_norm'].apply(replace_multiple_spaces)

    # Использование регулярного выражения для замены любого количества табов на один пробел
    def replace_tabs_with_space(input_string):
        return re.sub('\t+', ' ', input_string)

    jobs_df['job_name_norm'] = jobs_df['job_name_norm'].apply(replace_tabs_with_space)

    # Словарь для расшифровки сокращений
    abbreviation_dict = cfg["abbreviation_dict"]

    # Функция для расшифровки сокращений в названиях должностей
    def expand_abbreviations(job_name, abbr_dict=abbreviation_dict):
        job_name = " " + job_name + " "
        for abbr, full_form in abbr_dict.items():
            job_name = job_name.replace(" " + abbr + " ", " " + full_form + " ")
        return job_name.strip()

    jobs_df['job_name_norm'] = jobs_df['job_name_norm'].apply(expand_abbreviations)

    jobs_df['job_name_norm'] = jobs_df['job_name_norm'].str.replace(r"\s*-\s*", "-", regex=True)
    jobs_df['job_name_norm'] = jobs_df['job_name_norm'].str.replace(r"\s*/\s*", " / ", regex=True)

    # функция, которая преобразует римские цифры в арабские в строке
    def roman_to_arabic(input_string):
        def replace_roman_numerals(match):
            try:
                return str(roman.fromRoman(match.group().upper()))
            except roman.InvalidRomanNumeralError:
                return match.group()

        # Регулярное выражение для поиска римских цифр
        roman_numeral_pattern = r'\bM{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b'

        # Заменяем римские цифры на арабские
        res = re.sub(roman_numeral_pattern, replace_roman_numerals, input_string, flags=re.IGNORECASE)
        if res != input_string:
            print(f"roman_to_arabic() {input_string} -> {res}")
        return res

    jobs_df['job_name_norm'] = jobs_df['job_name_norm'].apply(roman_to_arabic)

    def drop_everything_after_first_digit(input_string: str) -> str:
        for index, char in enumerate(input_string):
            if char.isdigit():
                return input_string[:index]
        return input_string

    jobs_df['job_name_norm'] = jobs_df['job_name_norm'].apply(drop_everything_after_first_digit)

    def remove_reducant_spaces(text: str) -> str:
        return text.strip()

    jobs_df['job_name_norm'] = jobs_df['job_name_norm'].apply(remove_reducant_spaces)

    # Работает только с подключеним к сети
    def translate_all(input_string: str) -> str:
        def contains_english_letters(text):
            return bool(re.search(r'[a-zA-Z]', text))

        if contains_english_letters(input_string):
            translate = ts.translate_text(input_string, translator='bing', from_language="en", to_language="ru").lower()
            return translate
        else:
            return input_string

    if cfg['apply_eng_to_ru_translation']:
        jobs_df['job_name_norm'] = jobs_df['job_name_norm'].apply(translate_all)

    pd.set_option('display.max_rows', 10)

    is_spell_check_activated = cfg["activate_spell_check"]
    if is_spell_check_activated:
        spell = Speller("ru")

        # spell.nlp_data.update()

        def correct(text: str) -> str:
            res = str(spell(text))
            if res != text:
                print("autocorrect corrected " + text + " -> " + res)
            return res

        jobs_df['job_name_norm'] = jobs_df['job_name_norm'].apply(correct)

    from collections import Counter

    words = jobs_df['job_name_norm'].str.split().explode()

    canon_perc = cfg["canon_job_names_top_percentile"]
    apply_to_perc = cfg['try_to_find_canon_in_bottom_percentile']
    if canon_perc < 0 or canon_perc > 100:
        print("WARNING: Wrong canon_job_names_top_percentile value in config.")
    if apply_to_perc < 0 or apply_to_perc > 100:
        print("WARNING: Wrong try_to_find_canon_in_bottom_percentile value in config.")

    # Нет смысла в пересечении нижнего и верхнего списка
    if apply_to_perc + canon_perc > 100:
        print(
            "WARNING: canon_job_names_top_percentile and try_to_find_canon_in_bottom_percentile values in config overlap.")

    value_counts = jobs_df['job_name_norm'].value_counts()

    top_quantive_value = 1 - canon_perc / 100
    bottom_quantive_value = apply_to_perc / 100

    # Определение порога для верхнего и нижнего n-процентилей
    top_percent_threshold = value_counts.quantile(top_quantive_value)
    bottom_percent_threshold = value_counts.quantile(bottom_quantive_value)

    # Получение значений для верхнего и нижнего n-процентилей
    bottom_percent_values = value_counts[value_counts <= bottom_percent_threshold].index
    top_percent_values = set(value_counts[value_counts >= top_percent_threshold].index)

    # Список строк, которые нужно удалить из top_percent_values
    strings_to_remove = cfg["words_to_not_consider_as_canon_job_name"] + [" ", ""]

    # Удаление строк из top_10_percent_values
    top_percent_values -= set(strings_to_remove)

    print("Rows taken into top-percentile values: ", len(top_percent_values))
    print("Rows taken into bottom-percentile values: ", len(bottom_percent_values))

    # Функция для проверки, содержит ли строка из нижнего процентиля все слова из какой-либо строки верхнего процентиля
    def contains_all_words_from_top(bottom_value, top_values) -> list[str]:
        all_matches = []
        bottom_words = set(bottom_value.split())
        for top_value in top_values:
            top_words = set(top_value.split())
            if top_words.issubset(bottom_words):
                all_matches.append(top_value)

        return all_matches

    words_to_explicitly_check = cfg["words_to_never_discard"]

    # Итерация по строкам нижнего процентиля
    for bottom_value in bottom_percent_values:
        all_matches = contains_all_words_from_top(bottom_value, top_percent_values)
        if all_matches:
            for i, match in enumerate(all_matches):
                for word in words_to_explicitly_check:
                    if word in bottom_value and word not in match:
                        all_matches[i] = word + " " + match

        if not all_matches:
            continue

        # Выбрать лучший match из возможных (пока что по длинне строки)
        all_matches = sorted(all_matches, key=len, reverse=True)
        best_match = all_matches[0]
        if bottom_value != best_match:
            print(bottom_value + " -> " + best_match)
            print(all_matches)
            jobs_df['job_name_norm'] = jobs_df['job_name_norm'].replace(bottom_value, best_match)

    # Словарь для перевода названий месяцев на русском языке
    months = {
        'Январь': 1, 'Февраль': 2, 'Март': 3, 'Апрель': 4, 'Май': 5, 'Июнь': 6,
        'Июль': 7, 'Август': 8, 'Сентябрь': 9, 'Октябрь': 10, 'Ноябрь': 11, 'Декабрь': 12,
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }

    # Функция для преобразования строки в дату
    def convert_to_date(date_str):
        if date_str is None or date_str == 'None' or date_str.lower() == 'currently':
            return None
        else:
            month_str, year_str = date_str.split()
            month = months.get(month_str, 1)  # Значение по умолчанию - январь
            year = int(year_str)
            return datetime(year, month, 1)

    # Преобразование столбцов start и end в даты
    jobs_df['start_ts'] = jobs_df['start'].apply(convert_to_date)
    jobs_df['end_ts'] = jobs_df['end'].apply(convert_to_date)

    # Функция для расчета продолжительности работы в месяцах
    def calculate_duration(row):
        start_date = row['start_ts']
        end_date = row['end_ts'] if row['end_ts'] is not None else datetime.now()
        return (end_date.year - start_date.year) * 12 + end_date.month - start_date.month

    # Расчет продолжительности работы для каждой записи
    jobs_df['duration_months'] = jobs_df.apply(calculate_duration, axis=1)

    # Оптимизация скрипта расчета стажа

    # Сортировка df_jobs по person_id и дате начала работы
    jobs_df = jobs_df.sort_values(by=['person_id', 'start_ts'])

    # Инициализация словаря для хранения накопленного стажа для каждого person_id
    accumulated_experience = {}

    last_job_start_dict = {}

    last_added_exp = {}

    # Расчет стажа в одном проходе
    def calculate_optimized_experience(row):
        person_id = row['person_id']
        start_date: datetime = row['start_ts']
        experience = accumulated_experience.get(person_id, 0)
        last_job_start = last_job_start_dict.get(person_id, None)
        if experience < 0:
            print("Expirience is negative...")
        assuming_approx_prev_job_start = start_date - relativedelta(months=experience)

        # Проверка на вторую одновременную работу
        if last_job_start is not None and assuming_approx_prev_job_start < last_job_start:
            # Расчет разницы между датами
            difference = relativedelta(start_date, last_job_start)
            # Преобразование разницы в общее количество месяцев
            total_months = difference.years * 12 + difference.months
            if total_months < 0:
                print("TM is negative...")
            exp_to_substract = last_added_exp.get(person_id, 0)
            experience += total_months - exp_to_substract

        # Обновление накопленного стажа для person_id
        accumulated_experience[person_id] = experience + row['duration_months']
        last_added_exp[person_id] = row['duration_months']
        last_job_start_dict[person_id] = start_date
        return experience

    jobs_df['experience_at_start'] = jobs_df.apply(calculate_optimized_experience, axis=1)

    # Вывод результатов
    # print(jobs_df[['start', 'end', 'duration_months', 'experience_at_start']])

    jobs_df['cluster'] = np.where(jobs_df['experience_at_start'] < 6, -2, 0)

    for index, row in jobs_df.iterrows():
        if row['cluster'] == -2 and contains_all_words_from_top(row['job_name_norm'], top_percent_values):
            jobs_df.at[index, 'cluster'] = 0

    # Удаление стоп-слов из нормализованного представления наименования профессии

    # Скачивание списка стоп-слов
    nltk.download('stopwords')
    nltk.download('punkt')

    stop_words = set(stopwords.words('russian'))

    # Load additional stop words from the file stopwords-ru.json
    ru_additional_stop_words_file = '../stop-words/stopwords-ru.json'
    with open(ru_additional_stop_words_file, 'r', encoding='utf-8') as f:
        ru_additional_stop_words_file = set(json.load(f))

    # Combine the NLTK stopwords and additional words to exclude
    stop_words = stop_words.union(ru_additional_stop_words_file)

    stop_words -= set(words_to_explicitly_check)

    def remove_stop_words(text):
        # Разбиение текста на слова
        word_tokens = word_tokenize(text)

        # Фильтрация стоп-слов
        filtered_text = [word for word in word_tokens if not word in stop_words]

        # Сборка обратно в строку
        res = ' '.join(filtered_text)

        if res != text:
            print("Убраны стоп-слова: ", text, " -> ", res)

        return res

    # Применение функции к каждой строке в столбце DataFrame
    jobs_df['job_name_norm'] = jobs_df['job_name_norm'].apply(remove_stop_words)

    is_lemmatization_turned_on = cfg['lemmatize_job_names']
    if is_lemmatization_turned_on:
        def lemmatize_text(text):
            morph = pymorphy2.MorphAnalyzer()
            words = text.split()  # Разбиение текста на слова
            lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
            return ' '.join(lemmatized_words)

        # Применение функции к столбцу DataFrame
        jobs_df['job_name_norm_lemmatized'] = jobs_df['job_name_norm'].apply(lemmatize_text)

    # Установка значения -3 в столбце 'cluster' для строк, где длина 'job_name_norm' меньше 5
    jobs_df.loc[jobs_df['job_name_norm'].str.len() < 5, 'cluster'] = -3

    print("Убрано из рассмотрения строк по причине пустоты/недостаточной длинны имени профессии: ",
          (jobs_df['cluster'] == -3).sum())

    # Подключение к базе данных для сохранения сниппета
    db_path_norm = "../db/normalized_data.db"
    conn_norm = sqlite3.connect(db_path_norm)

    jobs_df.to_sql(name='jobs', con=conn_norm, if_exists='replace', index=False)
