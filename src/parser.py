from bs4 import BeautifulSoup
import re
import sqlite3
import glob


def parse():
    database = "../db/data.db"
    resumes = glob.glob("../data/resumes/*.html")

    def insert_person_data(conn, data):
        sql = ''' INSERT INTO person (age, gender, town, key_profession, specialtys, key_skills, months_of_exp)
                  VALUES (?, ?, ?, ?, ?, ?, ?) '''
        cur = conn.cursor()
        cur.execute(sql, data)
        conn.commit()
        return cur.lastrowid

    all_resumes = []
    for resume in resumes:
        with open(resume, 'r', encoding='utf-8') as file:
            html_content = file.read()
        all_resumes.append(BeautifulSoup(html_content, 'html.parser'))

    for i in range(len(all_resumes)):
        soup = all_resumes[i]

        # -------------------------
        # Пол, возраст, город
        gender_element = soup.find('span', {'data-qa': 'resume-personal-gender'})
        gender = gender_element.text if gender_element else None

        age_element = soup.find('span', {'data-qa': 'resume-personal-age'})
        age_text = age_element.text if age_element else None

        city_element = soup.find('span', {'data-qa': 'resume-personal-address'})
        city = city_element.text if city_element else None

        # Используем регулярные выражения для извлечения числа из строки возраста
        if age_text:
            age = int(re.search(r'\d+', age_text).group())
        else:
            age = None

        # -------------------------
        # Специализации
        specializations_elements = soup.find_all('li', class_='resume-block__specialization')

        specializations = [element.text.strip() for element in specializations_elements]

        specializations = ', '.join(specializations)

        # -------------------------
        # Опыт работы
        experience_elements = soup.find_all('span', class_='resume-block__title-text_sub')[0].find_all('span')
        if not experience_elements:
            experience_elements = soup.find_all('span', class_='resume-block__title-text_sub')[1].find_all('span')

        years = 0
        months = 0

        for element in experience_elements:
            text = element.text.strip()
            # Проверяем, содержит ли текст информацию о годах
            if 'лет' in text or 'год' in text:
                years = int(re.search(r'\d+', text).group())

            # Проверяем, содержит ли текст информацию о месяцах
            elif 'месяц' in text or 'месяцев' in text:
                months = int(re.search(r'\d+', text).group())

        total_months = years * 12 + months

        # -------------------------
        # Ключевые навыки
        if "Ключевые навыки" in html_content:
            # Находим блок с ключевыми навыками
            skills_block = soup.find('div', class_='bloko-tag-list')

            # Находим все элементы с ключевыми навыками
            skill_elements = skills_block.find_all('div', {'data-qa': 'bloko-tag bloko-tag_inline'})

            # Извлекаем текст каждого навыка
            skills = [skill.find('span', {'data-qa': 'bloko-tag__text'}).text.strip() for skill in skill_elements]

            skills = ', '.join(skills)
        else:
            skills = None

        # -------------------------
        # Основная профессия
        profession_element = soup.find('span', class_='resume-block__title-text')
        profession = profession_element.text.strip()

        # -------------------------
        # Запись в БД
        conn = sqlite3.connect(database)

        data = (age, gender, city, profession, specializations, skills, total_months)
        insert_person_data(conn, data)

        conn.close()

    def process_html(html_content, soup, person_id):
        is_high = 1 if "Высшее образование" in html_content else 0
        education_blocks = soup.find_all('div', {'data-qa': 'resume-block-education-item'})

        if not education_blocks:
            # Обработка случаев, когда блоки образования отсутствуют
            return [(person_id, None, None, is_high)]

        education_data = []
        for block in education_blocks:
            university_element = block.find('div', {'data-qa': 'resume-block-education-name'})
            university_name = university_element.text.strip() if university_element else 'Не указано'

            specialization_element = block.find('div', {'data-qa': 'resume-block-education-organization'})
            specialization_text = specialization_element.text.strip() if specialization_element else 'Не указано'

            education_data.append((person_id, university_name, specialization_text, is_high))

        return education_data

    with sqlite3.connect(database) as conn:
        cursor = conn.cursor()
        all_data = []

        for i in range(len(all_resumes)):
            soup = all_resumes[i]
            education_data = process_html(html_content, soup, i + 1)
            all_data.extend(education_data)
        cursor.executemany("INSERT INTO education (person_id, name, specialitys, is_high_edu) VALUES (?, ?, ?, ?)",
                           all_data)
        conn.commit()

    def insert_job_data(conn, data):
        sql = ''' INSERT INTO jobs (job_name, person_id, desc, start, end)
                  VALUES (?, ?, ?, ?, ?) '''
        cur = conn.cursor()
        cur.execute(sql, data)
        conn.commit()
        return cur.lastrowid

    for i in range(len(all_resumes)):
        soup = all_resumes[i]

        # Найти все блоки с опытом работы
        experience_blocks = soup.find_all("div",
                                          class_="bloko-column bloko-column_xs-4 bloko-column_s-2 bloko-column_m-2 "
                                                 "bloko-column_l-2")
        experience_blocks2 = soup.find_all("div",
                                           class_="bloko-column bloko-column_xs-4 bloko-column_s-6 bloko-column_m-7 "
                                                  "bloko-column_l-10")

        parsed_dates = []
        for block in experience_blocks:
            # Извлекаем текст и проверяем наличие разделителя дат
            text = block.get_text(separator=" ", strip=True)

            if " — " in text:
                # Разделяем текст на дату начала и дату окончания
                start_date, end_date = text.split(" — ")
                # Очищаем дату окончания от дополнительной информации
                end_date = " ".join(end_date.split()[:2])
            else:
                # Если разделитель дат отсутствует, устанавливаем дату начала и None для даты окончания
                start_date = text
                end_date = None
            if end_date is not None:
                if end_date == 'по настоящее' or "currently" in end_date:
                    end_date = None
                parsed_dates.append((start_date, end_date))

        # print(parsed_dates)

        job_descriptions = []
        for block in experience_blocks2:
            # Название профессии
            title_element = block.find("div", {"data-qa": "resume-block-experience-position"})
            job_title = title_element.get_text(strip=True) if title_element else None

            # Описание профессии
            description_element = block.find("div", {"data-qa": "resume-block-experience-description"})
            job_description = description_element.get_text(strip=True) if description_element else None
            if job_title is not None and job_description is not None:
                job_descriptions.append((job_title, job_description))

        for dates, jobs in zip(parsed_dates, job_descriptions):
            data = (jobs[0], i + 1, jobs[1], dates[0], dates[1])
            conn = sqlite3.connect(database)
            insert_job_data(conn, data)
            conn.close()
