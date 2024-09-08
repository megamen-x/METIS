import asyncio
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardButton, CallbackQuery, FSInputFile
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.filters.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
# from aiogram.client.telegram import TelegramAPIServer
# from aiogram.client.session.aiohttp import AiohttpSession
from database import Database

import torch
import torchaudio
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessor as NeMoAudioToMelSpectrogramPreprocessor,
)
from nemo.collections.asr.parts.preprocessing.features import (
    FilterbankFeaturesTA as NeMoFilterbankFeaturesTA,
)
import numpy as np
from pyannote.audio import Pipeline
from pydub import AudioSegment
import librosa
import soundfile as sf
from tinytag import TinyTag

from emaling import send_bulk_email
from pattern import fill_decrypton, fill_official, fill_unofficial
from docx_pdf import convert_words_to_pdfs, add_encryption
from punctuation_spell import update_punctuation
from infer import get_information
from pathlib import Path
from io import BytesIO
from typing import List, Tuple
import os
from dotenv import load_dotenv
from shutil import rmtree
import locale

load_dotenv()
locale.getpreferredencoding = lambda: "UTF-8"

# Define constants
BATCH_SIZE = 10

# Custom class for filterbank features
class FilterbankFeaturesTA(NeMoFilterbankFeaturesTA):
    def __init__(self, mel_scale: str = "htk", wkwargs=None, **kwargs):
        if "window_size" in kwargs:
            del kwargs["window_size"]
        if "window_stride" in kwargs:
            del kwargs["window_stride"]

        super().__init__(**kwargs)

        self._mel_spec_extractor: torchaudio.transforms.MelSpectrogram = (
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self._sample_rate,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=kwargs["nfilt"],
                window_fn=self.torch_windows[kwargs["window"]],
                mel_scale=mel_scale,
                norm=kwargs["mel_norm"],
                n_fft=kwargs["n_fft"],
                f_max=kwargs.get("highfreq", None),
                f_min=kwargs.get("lowfreq", 0),
                wkwargs=wkwargs,
            )
        )

# Custom class for audio preprocessing
class AudioToMelSpectrogramPreprocessor(NeMoAudioToMelSpectrogramPreprocessor):
    def __init__(self, mel_scale: str = "htk", **kwargs):
        super().__init__(**kwargs)
        kwargs["nfilt"] = kwargs["features"]
        del kwargs["features"]
        self.featurizer = (
            FilterbankFeaturesTA(  # Deprecated arguments; kept for config compatibility
                mel_scale=mel_scale,
                **kwargs,
            )
        )

class Diarization:
    def __init__(self, start, stop, speaker) -> None:
        self.start = start
        self.stop = stop
        self.speaker = speaker

# Set device (CPU or GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load ASR model
model = EncDecRNNTBPEModel.from_config_file("./rnnt_model_config.yaml")
ckpt = torch.load("./rnnt_model_weights.ckpt", map_location="cpu")
model.load_state_dict(ckpt, strict=False)
model.eval()
model = model.to(device)

# Load voice activity detection pipeline
pipeline = Pipeline.from_pretrained(
        "pyannote/voice-activity-detection", use_auth_token=os.getenv('HF_TOKEN')
    )
pipeline = pipeline.to(torch.device(device))

# Load diarization model
pipeline_diarization = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.getenv('HF_TOKEN'))
pipeline_diarization.to(torch.device(device))

# Function to convert AudioSegment to numpy array
def audiosegment_to_numpy(audiosegment: AudioSegment) -> np.ndarray:
    """Convert AudioSegment to numpy array."""
    samples = np.array(audiosegment.get_array_of_samples())
    if audiosegment.channels == 2:
        samples = samples.reshape((-1, 2))

    samples = samples.astype(np.float32, order="C") / 32768.0
    return samples

# Function to convert audio to WAV format with 16kHz sampling rate
def convert_to_wav_16k(input_file: str, output_file: str):
  """
  Converts any audio file to WAV format with a sampling rate of 16000 Hz.

  Args:
    input_file: Path to the input audio file (any format supported by librosa).
    output_file: Path to save the output WAV file.
  """
  try:
    # Load the audio file using librosa
    y, sr = librosa.load(input_file, sr=None)  # Load with original sampling rate

    # Resample to 16kHz if necessary
    if sr != 16000:
      y = librosa.resample(y, orig_sr=sr, target_sr=16000)

    # Save as WAV file with 16kHz sampling rate
    sf.write(output_file, y, 16000, subtype='PCM_16') 

  except Exception as e:
    print(f"Error converting file: {e}")

# Function to segment audio into chunks for ASR
def segment_audio(
    audio_path: str,
    pipeline: Pipeline,
    max_duration: float = 22.0,
    min_duration: float = 10.0,
    new_chunk_threshold: float = 0.2,
) -> Tuple[List[np.ndarray], List[List[float]]]:
    
    # Prepare audio for pyannote vad pipeline
    preprocessed_audio_path = '/'.join(audio_path.split('/')[:-1]) + '/preprocessed_audio.wav'
    convert_to_wav_16k(audio_path, preprocessed_audio_path)
    audio = AudioSegment.from_wav(preprocessed_audio_path)
    audio_bytes = BytesIO()
    audio.export(audio_bytes, format='wav')
    audio_bytes.seek(0)

    # Process audio with pipeline to obtain segments with speech activity
    sad_segments = pipeline({"uri": "filename", "audio": audio_bytes})

    segments = []
    curr_duration = 0
    curr_start = 0
    curr_end = 0
    boundaries = []

    # Concat segments from pipeline into chunks for asr according to max/min duration
    for segment in sad_segments.get_timeline().support():
        start = max(0, segment.start)
        end = min(len(audio) / 1000, segment.end)
        if (
            curr_duration > min_duration and start - curr_end > new_chunk_threshold
        ) or (curr_duration + (end - curr_end) > max_duration):
            audio_segment = audiosegment_to_numpy(
                audio[curr_start * 1000 : curr_end * 1000]
            )
            segments.append(audio_segment)
            boundaries.append([curr_start, curr_end])
            curr_start = start

        curr_end = end
        curr_duration = curr_end - curr_start

    if curr_duration != 0:
        audio_segment = audiosegment_to_numpy(
            audio[curr_start * 1000 : curr_end * 1000]
        )
        segments.append(audio_segment)
        boundaries.append([curr_start, curr_end])

    return segments, boundaries

# Function to format time in HH:MM:SS format
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    full_seconds = int(seconds)

    if hours > 0:
        return f"{hours:02}:{minutes:02}:{full_seconds:02}"
    else:
        return f"{minutes:02}:{full_seconds:02}"

# Function to perform ASR inference on the audio file
def audio_inference(file_path: str) -> str:
    preprocessed_audio_path = '/'.join(file_path.split('/')[:-1]) + '/preprocessed_audio.wav'
    convert_to_wav_16k(file_path, preprocessed_audio_path)
    waveform, sample_rate = torchaudio.load(preprocessed_audio_path)
    diarization = pipeline_diarization({"waveform": waveform, "sample_rate": sample_rate})

    # run the diarization pipeline on an audio file
    diarization = pipeline_diarization(preprocessed_audio_path)
    check_list = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if len(check_list) == 0:
            check_list.append(Diarization(turn.start, turn.end, speaker))
        elif speaker != check_list[-1].speaker:
            check_list.append(Diarization(turn.start, turn.end, speaker))
        else:
            check_list[-1].stop = turn.end
    postprocessing_list = []
    for el in check_list:
        if el.stop - el.start > 0.5:
            if len(postprocessing_list) == 0:
                postprocessing_list.append(el)
            elif el.speaker == postprocessing_list[-1].speaker:
                postprocessing_list[-1].stop = el.stop
            else:
                postprocessing_list.append(el)
    speakers = set()
    for el in postprocessing_list:
        speakers.add(el.speaker)
     # run the ASR pipeline on an audio file
    segments, boundaries = segment_audio(preprocessed_audio_path, pipeline)
    transcriptions = model.transcribe(segments, batch_size=BATCH_SIZE)[0]
    full_text = f'{postprocessing_list[0].speaker}:({format_time(postprocessing_list[0].start)}-{format_time(postprocessing_list[0].stop)})\n'
    asr_text = ''
    for transcription, boundary in zip(transcriptions, boundaries):
        if (boundary[0] >= postprocessing_list[0].start - 0.5) and (boundary[1] <= postprocessing_list[0].stop + 0.5):
            full_text += transcription + '\n'
        else:
            transcription_list = transcription.split(' ')
            if (postprocessing_list[0].stop - boundary[0]) < 0:
              postprocessing_list.pop(0)
              full_text += f'\n{postprocessing_list[0].speaker}:({format_time(postprocessing_list[0].start)}-{format_time(postprocessing_list[0].stop)})\n'
              full_text += transcription
            else: 
              formula = round( len(transcription_list) * ((postprocessing_list[0].stop - boundary[0]) / (boundary[1] - boundary[0])))
              part1 = ' '.join(transcription_list[:formula])
              part2 = ' '.join(transcription_list[formula:])
              full_text += part1
              postprocessing_list.pop(0)
              full_text += f'\n{postprocessing_list[0].speaker}:({format_time(postprocessing_list[0].start)}-{format_time(postprocessing_list[0].stop)})\n'
              full_text += part2

        asr_text +=f'{transcription}\n'
    return asr_text, full_text, len(speakers)

# Define states for the bot
class WorkStates(StatesGroup):
    DEFAULT = State()
    E_MAILING = State()
    TG_MAILING = State()
    SET_EMAIL = State()

# Main class for the Telegram bot
class Metis:

    def __init__(self, bot_token: str, db_path: str) -> None:
        """
        Initialize the Metis bot.

        Args:
            bot_token (str): The Telegram bot token.
            db_path (str): Path to the database file.
        """
        
        self.db = Database(db_path)
        self.bot = Bot(token=bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML)) 
        self.dp = Dispatcher()

        # Register handlers for different commands and states
        self.dp.message.register(self.start, Command("start"))
        self.dp.message.register(self.about_team, Command("team"))
        self.dp.message.register(self.start_mailing, Command('mailing'))
        self.dp.message.register(self.set_email, Command('set_email'))

        self.dp.message.register(self.asr_mod, WorkStates.DEFAULT)
        self.dp.message.register(self.tg_mailing, WorkStates.TG_MAILING)
        self.dp.message.register(self.e_mailing, WorkStates.E_MAILING)

        self.dp.callback_query.register(self.get_mailing_list, F.data.startswith('mailing'))
        self.dp.callback_query.register(self.get_extenstion, F.data.startswith('extention'))
    
    async def start_polling(self):
        """Start polling for updates from Telegram."""
        await self.dp.start_polling(self.bot) # Start polling for updates
    
    async def start(self, message: Message, state: FSMContext):
        """
        Handle the /start command.

        Creates a new user in the database, sets the state to DEFAULT,
        сreates a media directory for the user, and prompts for an audio message.

        Args:
            message (Message): The incoming message object.
            state (FSMContext): The current state of the conversation.
        """
        await state.set_state(WorkStates.DEFAULT)
        self.db.create_user(user_id=message.chat.id, username=message.from_user.username)
        if os.path.exists(f'media/{message.chat.id}'):
            rmtree(f'media/{message.chat.id}')
        os.makedirs(f'media/{message.chat.id}', exist_ok=True)
        await message.reply("Введите аудиосообщение")

    async def about_team(self, message: Message):
        """
        Handle the /team command.

        Sends a photo of the team and a description message.

        Args:
            message (Message): The incoming message object.
        """
        await self.bot.send_photo(photo=FSInputFile("megamen team.jpg"), chat_id=message.chat.id)
        await self.bot.send_message(
            chat_id=message.chat.id,
            text="""Мы, команда megamen, является частыми участниками хакатонов разного уровня. 
Наши проекты остаются в памяти надолго за счет высокой точности, скорости и дизайна. 
Надеемся, что данный продукт ваш будет полезен."""
        )

    async def asr_mod(self, message: Message):
        """
        Process audio messages and generate meeting documents.

        This method handles audio, voice, and document messages, performs speech recognition,
        and generates various meeting documents.

        Args:
            message (Message): The incoming message object.
        """

        # Determine the file_id based on the message type
        chat_id = message.chat.id
        if message.audio: 
            file_id = message.audio.file_id
        elif message.voice:
            file_id = message.voice.file_id
        elif message.document:
            file_id = message.document.file_id

        # Get the file information and download it
        file = await self.bot.get_file(file_id)
        file_path = file.file_path
        new_file_path = f"media/{chat_id}/audio{Path(file_path).suffix}"
        await self.bot.download_file(file_path, new_file_path)

        # Perform speech recognition on the audio file
        asr_text, decryption, count_speakers = audio_inference(new_file_path)
        audio_duration = TinyTag.get(f"media/{chat_id}/preprocessed_audio.wav").duration
        audio_duration = format_time(audio_duration)
        asr_text = update_punctuation(asr_text)
        decryption = update_punctuation(decryption)
        decryption = decryption.replace(f'SPEAKER ', '\nСпикер')
        decryption = decryption[1:]
        print(asr_text, count_speakers)
        text = get_information(asr_text)
        meeting_name = text.split("Название совещания: ")[1].split("\n")[0]
        question_title = text.split("Какие задачи были рассмотрены на совещании: ")[1].split("\n")[0].split(",")
        name_list = []
        position_list = []
        for item in text.split("ФИО и должности всех участников совещания: ")[1].split("\n")[0].split(", "):
            name_list.append(item.split(" - ")[0])
            position_list.append(item.split(" - ")[1])
        # Generate different types of meeting documents
        document_1 = fill_official(meeting_name=meeting_name,
                                   position_list=position_list,
                                   name_list=name_list,
                                   question_title=question_title,
                                   file_name=f'media/{chat_id}/Official_protocol.docx')
        document_2 = fill_unofficial(meeting_name=meeting_name,
                                   position_list=position_list,
                                   name_list=name_list,
                                   question_title=question_title,
                                   audio_duration=audio_duration, 
                                   file_name=f'media/{chat_id}/Unofficial_protocol.docx')
        document_3 = fill_decrypton(full_text=decryption, 
                                    file_name=f'media/{chat_id}/Transcript.docx')

        # Store the generated documents in the database
        self.db.create_meeting_text(user_id=chat_id, decryption=document_1, off_doc=document_2, unoff_doc=document_3)
        
        # Create an inline keyboard for file format selection
        kb = InlineKeyboardBuilder()
        kb.add(InlineKeyboardButton(chat_id=chat_id, text='DOCX', callback_data='extention:docx:no'))
        kb.add(InlineKeyboardButton(chat_id=chat_id, text='PDF без пароля', callback_data='extention:pdf:no'))
        kb.add(InlineKeyboardButton(chat_id=chat_id, text='PDF с паролем', callback_data='extention:pdf:yes'))
        markup = kb.as_markup()
        await self.bot.send_message(chat_id=chat_id, text='Выберите расширение файлов', reply_markup=markup)

    def send_files(self, user_id: int, extention: str, is_encrypt: bool=False):
        """
        Prepare and send meeting documents to the user.

        This method retrieves the meeting documents from the database, converts them to the
        specified format, and prepares them for sending as a media group.

        Args:
            user_id (int): The ID of the user requesting the files.
            extention (str): The desired file extension ('docx' or 'pdf').
            is_encrypt (bool, optional): Whether to encrypt PDF files. Defaults to False.

        Returns:
            list: A list of dictionaries representing the media group to be sent.
        """
        
        media_group = []
        meeting_documents = self.db.get_meeting_text(user_id=user_id)
        password = ''

        # Convert documents to PDF if necessary and handle encryption
        if extention == 'pdf':
            print('send_files', *meeting_documents)
            meeting_documents = convert_words_to_pdfs(meeting_documents)
            if is_encrypt:
                password = add_encryption(meeting_documents)

        # Prepare the media group with the document files
        for file_path in meeting_documents:
            file = FSInputFile(file_path)
            media_group.append({
            'type': 'document',
            'media': file,
        })
            
        # Add a caption to the last document in the media group
        media_group[2]['caption'] = f'Транскрибированное совещание в формате {extention}'
        if is_encrypt:
            media_group[2]['caption'] += f'\nПароль ко всем файлам: {password}'
        return media_group

    async def get_extenstion(self, callback:CallbackQuery):
        """
        Handle callback query to get file extension and encryption status, then send corresponding media files to the user.

        This method processes the callback data to determine the file extension and whether the file should be encrypted,
        then sends the appropriate media files to the user.

        Args:
            callback (CallbackQuery): The callback query containing data about file extension and encryption status.
        """
        
        # Extract file extension and encryption status from callback data
        chat_id = callback.message.chat.id
        extention, is_encrypt = callback.data.split(':')[1:]
        is_encrypt = 1 if is_encrypt == 'yes' else 0
        media_group = self.send_files(chat_id, extention, is_encrypt)
        # Send the media files to the user
        await self.bot.send_media_group(chat_id=chat_id,
                                media=media_group)

    async def set_email(self, message: Message):
        """
        Prompt the user to input their Gmail email and application-specific password.

        This method sends a message to the user requesting their Gmail email and application-specific password,
        providing instructions on how to obtain the password.

        Args:
            message (Message): The incoming message object containing chat information.
        """
        await self.bot.send_message(chat_id=message.chat.id, text='Пожалуйста, введите через пробел ваш gmail почту и пароль приложения (для второго необходимо перейти по ссылке https://myaccount.google.com/ зайти в раздел безопасность и найти пункт пароли приложений)')

    async def update_email(self, message: Message):
        """
        Update the user's email and application-specific password in the database.

        This method extracts the email and application-specific password from the message text,
        updates the user's email and password in the database, and notifies the user.

        Args:
            message (Message): The incoming message object containing the user's email and application-specific password.
        """
        # Extract and strip the email and app key from the message text
        email, app_key = [el.strip() for el in message.text.split()]
        self.db.set_user_email(user_id=message.chat.id, new_email=email, new_app_key=app_key)
        
        # Notify the user that their email has been updated
        await self.bot.send_message(chat_id=message.chat.id, text='Ваша почта обновлена')

    async def start_mailing(self, message: Message, state=FSMContext):
        """
        Initiate the mailing process by asking the user to choose between Telegram and Email mailing options.

        This method presents the user with a choice of mailing methods (Telegram or Email) using an inline keyboard.

        Args:
            message (Message): The incoming message object containing chat information.
            state (FSMContext): The Finite State Machine context for managing user states.
        """
         
        # Create an inline keyboard
        kb = InlineKeyboardBuilder()
        kb.add(InlineKeyboardButton(text='ТГ рассылка', callback_data='mailing:tg'))
        kb.add(InlineKeyboardButton(text='Email рассылка', callback_data='mailing:email'))
        markup = kb.as_markup()
        await self.bot.send_message(chat_id=message.chat.id, text='Выберите метод рассылки', reply_markup=markup)

    async def get_mailing_list(self, callback: CallbackQuery, state=FSMContext):
        """
        Handler for mailing list selection.

        This method is triggered when the user selects either Telegram or email mailing option.
        It prompts the user to enter a list of usernames or email addresses for mailing.

        Args:
            callback (CallbackQuery): The callback query object containing the user's selection.
            state (FSMContext, optional): The state context for managing the bot's state.
        """

        method = callback.data.split(':')[1]
        if method == 'tg':
             # If the user selects Telegram mailing
            await self.bot.send_message(chat_id=callback.message.chat.id, text='Укажите список пользователей (ники), которым хотите переслать сообщение. Пример ввода: Agar1us, whatislove7, @asitcomes, @al_goodini')
            await state.set_state(WorkStates.TG_MAILING)
        else:
            # If the user selects email mailing
            await self.bot.send_message(chat_id=callback.message.chat.id, text='Укажите список почт, на которые необходимо прислать документ. Пример ввода: sample@gmail.com, kasanta@yandex.ru,katamaran@list.ru')
            await state.set_state(WorkStates.E_MAILING)

    async def tg_mailing(self, message: Message):
        """
        Handler for Telegram mailing.

        This method is triggered when the user provides a list of usernames for Telegram mailing.
        It sends the meeting documents to the specified usernames.

        Args:
            message (Message): The message object containing the list of usernames.
        """

        chat_id = message.chat.id
        success_count = 0

        # Extract usernames from the message text
        users = [el.strip().replace('@', '') for el in message.text.split(',')]  
        for username in users:
            try:
                # Get the user ID from the database
                user_id = self.db.get_user(username=username).user_id  
                await self.bot.send_media_group(
                    chat_id=user_id, 
                    media=self.send_files(user_id=user_id, extention='pdf', is_encrypt=False)) 
                success_count += 1
            except Exception as e:
                await self.bot.send_message(chat_id=chat_id,
                    text=f"Не удалось отправить сообщение {username}")
        await self.bot.send_message(chat_id=chat_id, 
                text=f"Рассылка завершена. Сообщение удалось успешно разослать {success_count} из {len(users)} пользователям.")

    async def e_mailing(self, message:Message):
        """
        Handler for email mailing.

        This method is triggered when the user provides a list of email addresses for mailing.
        It sends the meeting documents as attachments to the specified email addresses.

        Args:
            message (Message): The message object containing the list of email addresses.
        """
        
        # Get the user object from the database
        user = self.db.get_user(username=message.from_user.username)
        if isinstance(user, str) or not user.email:
            # Use a default email and password
            sender_email = os.getenv('EMAIL')
            sender_password = os.getenv('EMAIL_PASSWORD')
        else:
            # Use the user's app key as the password and email
            sender_email = user.email
            sender_password = user.app_key
        subject = "Запись совещания"
        email_message = "В приложении находятся файлы с описанием совещания"
        attachment_paths = self.db.get_meeting_text(user_id=user.user_id)
        send_bulk_email(sender_email, sender_password, message.text, subject, email_message, attachment_paths)
        await self.bot.send_message(chat_id=user.user_id, text='Файлы были отправлены.')


# The bot initialization function
def main() -> None:
    """
    Main function to initialize and start the bot.

    Args:
        None

    Returns:
        None
    """

    bot = Metis(
        bot_token=os.getenv('BOT_TOKEN'),
        db_path=os.getenv('DB_PATH')
    )
    os.makedirs('media', exist_ok=True)

    # logout нужен для того, чтобы потом можно было перейти на локальный сервер (либо успеем на хаке, либо уже в дальнейшем)
    # is_logout = await bot.bot.log_out()
    # print(is_logout)
    
    asyncio.run(bot.start_polling())


if __name__ == '__main__':
    main()