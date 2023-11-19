# Используйте официальный образ Python как базовый
FROM python:3.11

# Установите рабочую директорию в контейнере
WORKDIR /usr/src/app

# Скопируйте файлы зависимостей и установите их
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Скопируйте ваш код в контейнер
COPY . .

# Укажите команду для запуска приложения
CMD ["python", "./src/main.py"]
