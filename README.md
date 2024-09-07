<a name="readme-top"></a>  
<img width="100%" src="https://github.com/megamen-x/METIS/blob/main/assets/pref_github.png" alt="megamen banner">
<div align="center">
  <p align="center">
  </p>

  <p align="center">
    <p></p>
    Создано <strong>megamen</strong>, совместно с <br /> <strong> Правительством Российской Федерации</strong>
    <br /><br />
    <a href="https://github.com/megamen-x/METIS/issues" style="color: black;">Сообщить об ошибке</a>
    ·
    <a href="https://github.com/megamen-x/METIS/discussions/1" style="color: black;">Предложить улучшение</a>
  </p>
</div>

**Содержание:**
- [Проблематика](#title1)
- [Описание решения](#title2)
- [Тестирование и запуск](#title3)
- [Обновления](#title4)

## <h3 align="start"><a id="title1">Проблематика</a></h3> 
Необходимо создать MVP ИИ-секретаря, работающего на сервере организации (или в облаке).

Ключевые функции программного модуля:
* обработка аудиозаписей совещаний:
  * генерация транскрипций;
  * обработка транскрипции с помощью LLM;
  * распознание и озаглавливание спикеров;
* формирование итогового протокола в заданном формате:
  * официальный протокол;
  * неофициальный протокол;
  * расшифровка встречи;
* формирование распределения задач на основе распознавания;
* формирование документов в разных форматах и возможностью установки пароля; 

<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>


## <h3 align="start"><a id="title2">Описание решения</a></h3>

**Machine Learning:**

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)

 - **Общая схема решения:**

<img width="100%" src="https://github.com/megamen-x/METIS/blob/main/assets/sheme-github.png" alt="megamen sheme">

 - **Использованные модели:**
    - **```ASR```**:
      - salute-developers/GigaAM;
    - **```Spell check```**:
      - kontur-ai/sbert_punc_case_ru;
    - **```VAD```**:
      - pyannote/voice-activity-detection;
    - **```LMM```**:
      - google/gemma-2-27b-it.


**Серверная часть**

[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)


<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>


## <h3 align="start"><a id="title3">Тестирование и запуск</a></h3> 

Данный репозиторий предполагает следующую конфигурацию тестирования решения:
  
  **```Telegram-bot + FastAPI + ML-models;```**

<details>
  <summary> <strong><i> Инструкция по запуску FastAPI-сервера:</i></strong> </summary>
  
  - В Visual Studio Code (**Windows-PowerShell activation recommended**) через терминал последовательно выполнить следующие команды:
  
    - Клонирование репозитория:
    ```
    git clone https://github.com/megamen-x/METIS.git
    ```
    - Создание и активация виртуального окружения (Протестировано на **Python 3.10.6**):
    ```
    cd ./ARTEMIS
    python -m venv .venv
    .venv\Scripts\activate
    ```
    - Уставновка зависимостей (при использовании **CUDA 12.4**):
    ```
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip3 install -r requirements.txt
    ```
    - Запуск чат-бота:
    ```
    python bot.py --bot_token={your_bot_token} --db_path={db_file_name}.db
    ```

</details> 


</br> 

**Аппаратные требования**

| Обеспечение | Требование |
| :----------- | :---------- |
| Платформа, ОС  | Windows (> 8.1), Linux (core > 5.15)    |
| Python | 3.10 or 3.11 (рекомендовано) |
| RAM  | 4 GB или более |
| Свободное место на диске | > 2 GB |
| GPU | NVIDIA RTX Ampere or Ada Generation GPU > 16 GB VRAM |

<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>


## <h3 align="start"><a id="title4">Обновления</a></h3> 

***Все обновления и нововведения будут размещаться здесь!***

<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>
