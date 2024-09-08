from docxtpl import DocxTemplate
from docxtpl import Listing
import jinja2
import pandas as pd
from datetime import date
import re


def auto_dock_of(meeting_info, position, name, question_title, question_binding, assignment_binding, speakers, resolved, disc_context, disc_time, contract_org, to_do, date_up_to, output_file):
    """
    Формирует документ по официальному шаблону

    Аргументы: 
        participants (list[dict]) - Участники собрания:
            position (str) - Должность участника;
            name (str) - ФИО участника;
        questions (list[dict]) - Вопросы собрания:
            question_title (str) - Название вопроса;
            speakers (list[str]) - Докладчики;
        solutions (list[dict]) - Принятые решения по каждому из вопросов:
            question_binding (str) - Привязка вопросов;
            resolved (str) - Принятое решение;
            disc_context (str) - Контекст обсуждения;
            disc_time (str) - Время обсуждения;
        assignments (list[dict]) - Поручения по итогам совещания:
            assignment_binding (str) - Привязка поручений;
            contract_org (str) - Организация-исполнитель;
            to_do (str) - Что нужно сделать;
            date_up_to (str) - Срок;
    """
    # participants
    participantsDF = pd.DataFrame({'position': position,'name': name,})
    # questions
    questionsDF = pd.DataFrame({'question_title': question_title, 
                                'speakers': speakers,})
    # solutions
    solutionsDF = pd.DataFrame({'question_binding': question_binding, 
                                'resolved': resolved,
                                'disc_context': disc_context, 
                                'disc_time': disc_time,})
    # assignments
    assignmentsDF = pd.DataFrame({'assignment_binding': assignment_binding, 
                                  'contract_org': contract_org, 
                                  'to_do': to_do, 
                                  'date_up_to': date_up_to,})
    # all context
    context = {
        "meeting_topic" : meeting_info[0],
        "meeting_date" : meeting_info[1],
        "participants" : participantsDF.to_dict(orient='records'),
        "n_rows_1": participantsDF.shape[0],
        "n_columns_1": participantsDF.shape[1],
        "questions" : questionsDF.to_dict(orient='records'),
        "solutions" : solutionsDF.to_dict(orient='records'),
        "assignments" : assignmentsDF.to_dict(orient='records'),
    }
    # document scan
    tpl = DocxTemplate('./pr_of_jinja2.docx')
    # render and save new document
    jinja_env = jinja2.Environment(autoescape=True)
    tpl.render(context, jinja_env)
    tpl.save(output_file)

def auto_dock_neof(meeting_info, position, name, question_title, question_binding, assignment_binding, speakers, resolved, disc_context, disc_time, contract_org, to_do, date_up_to, output_file):
    """
    Формирует документ по неофициальному шаблону

    Аргументы: 
        participants (list[dict]) - Участники собрания:
            position (str) - Должность участника;
            name (str) - ФИО участника;
        questions (list[dict]) - Вопросы собрания:
            question_title (str) - Название вопроса;
            speakers (list[str]) - Докладчики;
        solutions (list[dict]) - Принятые решения по каждому из вопросов:
            question_binding (str) - Привязка вопросов;
            resolved (str) - Принятое решение;
            disc_context (str) - Контекст обсуждения;
            disc_time (str) - Время обсуждения;
        assignments (list[dict]) - Поручения по итогам совещания:
            assignment_binding (str) - Привязка поручений;
            contract_org (str) - Организация-исполнитель;
            to_do (str) - Что нужно сделать;
            date_up_to (str) - Срок;
    """
    # participants
    participantsDF = pd.DataFrame({'position': position, 
                                    'name': name,})
    # questions
    questionsDF = pd.DataFrame({'question_title': question_title, 
                                'speakers': speakers,})
    # solutions
    solutionsDF = pd.DataFrame({'question_binding': question_binding, 
                                'resolved': resolved,
                                'disc_context': disc_context, 
                                'disc_time': disc_time,})
    # assignments
    assignmentsDF = pd.DataFrame({'assignment_binding': assignment_binding, 
                                  'contract_org': contract_org, 
                                  'to_do': to_do, 
                                  'date_up_to': date_up_to,})
    # all context
    context = {
        "meeting_topic" : meeting_info[0],
        "meeting_date" : meeting_info[1],
        "meeting_time" : meeting_info[2],
        "meeting_duration" : meeting_info[3],
        "meeting_goal" : meeting_info[4],
        "participants" : participantsDF.to_dict(orient='records'),
        "n_rows_1": participantsDF.shape[0],
        "n_columns_1": participantsDF.shape[1],
        "questions" : questionsDF.to_dict(orient='records'),
        "solutions" : solutionsDF.to_dict(orient='records'),
        "assignments" : assignmentsDF.to_dict(orient='records'),
    }
    # document scan
    tpl = DocxTemplate('./pr_neof_jinja2.docx')
    # render and save new document
    jinja_env = jinja2.Environment(autoescape=True)
    tpl.render(context, jinja_env)
    tpl.save(output_file)

def fill_official(meeting_name, position_list, name_list, question_title, file_name: str):
    meeting_info = [meeting_name, '12.07.2024', '14:33', '01:15', 'декларация цели встречи']
    question_binding = ['Название вопроса 1', 'Название вопроса 1']
    assignment_binding = ['Название вопроса 1', 'Название вопроса 2']
    speakers = ['Докладчик 1, докладчик 2', 'Докладчик 1, докладчик 2, докладчик 3', 'Докладчик 3']
    resolved = ['Принятое решение 1', 'Принятое решение 2']
    disc_context = ['краткий контекст обсуждения по достигнутому решению', 'краткий контекст обсуждения по достигнутому решению']
    disc_time = ['мм:сс', 'мм:сс']
    contract_org = ['Организация-исполнитель 1 (И.О. Фамилия руководителя) Организация-исполнитель 2 (И.О. Фамилия руководителя)', 'Организация-исполнитель 1 (И.О. Фамилия руководителя) Организация-исполнитель 2 (И.О. Фамилия руководителя) Организация-исполнитель 3 (И.О. Фамилия руководителя) ']
    to_do = ['Сделать то-то.', 'Сделать то-то.']
    date_up_to = ['1 сентября 2024 г.', '2 сентября 2024 г.']
    auto_dock_of(meeting_info, position_list, name_list, question_title, question_binding, assignment_binding, speakers, resolved, disc_context, disc_time, contract_org, to_do, date_up_to, file_name)
    return file_name

def fill_unofficial(meeting_name, position_list, name_list, question_title, audio_duration, file_name: str):
    today = date.today()
    today = today.strftime("%d.%m.%Y")
    meeting_info = [meeting_name, today, '', audio_duration, ' ']
    question_binding = ['Название вопроса 1', 'Название вопроса 1']
    assignment_binding = ['Название вопроса 1', 'Название вопроса 2']
    speakers = ['Докладчик 1, докладчик 2', 'Докладчик 1, докладчик 2, докладчик 3', 'Докладчик 3']
    resolved = ['Принятое решение 1', 'Принятое решение 2']
    disc_context = ['краткий контекст обсуждения по достигнутому решению', 'краткий контекст обсуждения по достигнутому решению']
    disc_time = ['мм:сс', 'мм:сс']
    contract_org = ['Организация-исполнитель 1 (И.О. Фамилия руководителя) Организация-исполнитель 2 (И.О. Фамилия руководителя)', 'Организация-исполнитель 1 (И.О. Фамилия руководителя) Организация-исполнитель 2 (И.О. Фамилия руководителя) Организация-исполнитель 3 (И.О. Фамилия руководителя) ']
    to_do = ['Сделать то-то.', 'Сделать то-то.']
    date_up_to = ['1 сентября 2024 г.', '2 сентября 2024 г.']
    auto_dock_neof(meeting_info, position_list, name_list, question_title, question_binding, assignment_binding, speakers, resolved, disc_context, disc_time, contract_org, to_do, date_up_to, file_name)
    return file_name


def fill_decrypton(full_text: str, file_name: str):
    """
    Формирует документ расшифровки

    Аргументы: 
        decryption (list[dict]) - Полная расшифровка:
    """
    context = {"decryption" : full_text}
    # document scan
    tpl = DocxTemplate('./pr_rash_jinja2.docx')
    # render and save new document
    jinja_env = jinja2.Environment(autoescape=True)
    tpl.render(context, jinja_env)
    tpl.save(file_name)
    return file_name
