import soundfile as sf
import requests
import yaml
import numpy as np
import telegram
from telegram.ext import *
import io
import librosa
from PIL import Image
import matplotlib.pyplot as plt
import os
from os import  listdir
import cv2
import tensorflow_hub as hub
import tensorflow as tf
from pytube import YouTube
from yt_dlp  import YoutubeDL
import yt_dlp
from inference import (build_classifier_speechbrain,
    build_label_decoder, build_name_decoder,
    main as recognize)



# Конфигурация
def read_config(filename = './config.yaml'):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config


# берет нужные кадрые из видео по ссылке
def exctract_cap(id_video, start=0, n=10, name = "frame"):
    """
    param: id_video это id видео из датасета (без пробелов)
    param: start - int говорит с какого кадра начать
    param: n - говорит сколько кадров взять
    param: name - будет сохранять картинки как name + str(i), где расстояния в кадрах от кадра start

    """
    url = "https://www.youtube.com/watch?v=" + str(id_video)
    ydl_opts = {}
    ydl = yt_dlp.YoutubeDL(ydl_opts)
    info_dict = ydl.extract_info(url, download=False)
    names = []
    formats = info_dict.get('formats', None)

    for f in formats:
        if f.get('format_note', None) == '360p':
            url = f.get('url', None)
            cap = cv2.VideoCapture(url)
            x = 0
            count = start
            cap.set(1, count)
            while x < n:
                ret, frame = cap.read()
                if not ret:
                    break
                filename = "content/" + name + str(x) + ".jpeg"
                x += 1
                names.append(filename)
                cv2.imwrite(filename, frame)
                # count+=300 #Skip 300 frames i.e. 10 seconds for 30 fps

            # cap.release()
    return names


def get_info(person_id, n, PATH, windows = 0):
    """
    :param person_id: айди человека, фото которого нужно вывести
    :param n: количество кадров, на которых присутствует данный человек
    :return: айди видео и массив [кадр,x,y,w, h]
    """

    lines = []
    text = []
    # D:\speech_project\hse_project\speaker_recognition_bot.py
    if ( not windows ):
        video_id = os.listdir(PATH + "txt/" + str(person_id))[0]
        f = os.listdir(PATH + "txt/" + str(person_id) + "/" + str(video_id))[0]
        file_value = open(PATH + "txt/" + str(person_id) + "/" + str(video_id) + "/" + str(f), 'r')
    else:
        video_id = os.listdir(PATH + "txt/" + str(person_id))[0]
        f = os.listdir(PATH + "txt/" + str(person_id) + "/" + str(video_id))[0]
        file_value = open(PATH  + "txt" + "/" + str(person_id) + "/" + str(video_id) + "/" + str(f), 'r')
    for line in file_value:
        lines.append(line)

    for i in range(7,7+n):
        text.append(lines[i].split())

    return video_id, text

#грузит фотку для создания картинок
def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]

    return img

#cоздает картинку
def art_face(main_img: str,minor_img:str, style_img = "const_img\style_4_picasso.jpeg"):
  model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
  content_image = load_image(main_img)
  # minor_image = load_image(minor_img)
  style_image = load_image(style_img)
  # stylized_image = model(tf.constant(content_image), tf.constant(minor_image))[0]
  stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

  file_name = 'content/result.jpeg'
  cv2.imwrite(file_name, cv2.cvtColor(np.squeeze(stylized_image)*255, cv2.COLOR_BGR2RGB))

  return file_name

#Проверяет есть ли двнный id локальной копии датасета
def check_id (id, path = "D:/speech_project/hse_project/txt"):
    """
    id - id звезды
    path - путь до файла где находятся все айдишники
    """
    a = listdir(path)
    if id in a:
        return True

    return False

# Берем из данных датасета квадрат лица
def face_by_data(lst, img):
    """
    lst - массив массивов из get_info
    img - путь к кадров с ютюба


    return - путь к сохранённому face
    """

    lst = lst[0]
    x,y,w,h = int(lst[1]), int(lst[2]), int(lst[3]), int(lst[4])
    img = np.asarray(Image.open(img))
    img = Image.fromarray(img[y:y + h,x : x + w, ])
    img.save("content/face_box.jpeg")
    return "content/face_box.jpeg"

#не обращайте внимание
def get_face_box(img_path, not_found_path='const_img/no_page_box.jpeg'):# должна была сама выделять лицо,
                        # но тут есть уникальный параметр который подбирает в зависимотси от размеров лица
  # Load the cascade
  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  # Read the input image
  img = cv2.imread(img_path)
  # Convert into grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # Detect faces
  faces = face_cascade.detectMultiScale(gray, 1.1, 10)
  file_name = "content/face_box.jpeg"
  if type(faces) is tuple and not faces:
    img_none = cv2.imread(not_found_path)
    cv2.imwrite(file_name, img_none)
  else:
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    cv2.imwrite(file_name, face)

  return file_name



#не обращайте внимание
def get_spectrogram(scale):
  FRAME_SIZE = 2048
  HOP_SIZE = 512
  S_scale = librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
  Y_scale = np.abs(S_scale) ** 2
  Y_log_scale = librosa.power_to_db(Y_scale)
  return Y_log_scale
#не обращайте внимание
def plot_spectrogram(Y, sr, hop_length = 512, y_axis="linear"):
    fig = plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis)
    file_name = 'content\spectro.jpeg'
    fig.savefig(file_name)
    return file_name



def proccess_audio(audio_bytes, target_sr = 16000):
    data, sr = sf.read(io.BytesIO(audio_bytes))

    audio = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    spectrogramm = get_spectrogram(audio)
    specto_file = plot_spectrogram(spectrogramm, target_sr, 512)


    global classifier
    global label_decoder
    global name_mapping

    label_id, score, embedding = recognize(classifier, audio, label_decoder)
    name_id = name_mapping.get(label_id, label_id)

    # Генерируем случайное изображение из numpy и отправляем его в качестве фото-ответа
    # img = (embedding/embedding.__abs__().max()) * 255
    # img = img.astype(np.uint8)
    # img = img[0].transpose()
    # img = Image.fromarray(spectrogramm)

    if check_id(label_id):
        v_id, n_frames = get_info(label_id, 1, "", 1)
        face = exctract_cap(v_id,0, 1)
        face = face_by_data(n_frames,face[0])
        result = art_face(face, specto_file)
    else:
        result = "const_img/no_in_dataset.jpg"
    # face = get_face_box(face[0])

    # img = Image.open('spectro.png') #рисует спектра грамму
    img = Image.open(result)
    bio = io.BytesIO()
    bio.name = 'image.png'
    img.save(bio, 'PNG')
    bio.seek(0)


    return bio, name_id, score


def clean_cash():
    dir = 'content/'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

# Функция для обработки аудиосообщений
async def audio_message_handler(update: telegram.Update, context: CallbackContext):
    clean_cash()
    audi_file_id = (update.message.audio or update.message.voice).file_id
    audio_file = await context.bot.get_file(audi_file_id)
    audio_bytes = await audio_file.download_as_bytearray()
    # Отправляем ответное сообщение

    await update.message.reply_text("аудио получено, обрабатываю...")

    img, name_id, score = proccess_audio(audio_bytes)

    await update.message.reply_text(f'Вы похожи на {name_id}%')
    await context.bot.send_photo(chat_id=update.message.chat_id, photo=img)
    # clean_cash()
    return

async def start_commmand_handler(update, context):
    await update.message.reply_text('Hello! Welcome to this voice2celeb bot!')

# Точка входа
def main(config: dict):

    # Создаем объект Updater и передаем ему токен бота

    application = Application.builder().token(config['api_token']).build()

    # Commands
    application.add_handler(CommandHandler('start', start_commmand_handler))
    application.add_handler(MessageHandler(filters.AUDIO, audio_message_handler))
    application.add_handler(MessageHandler(filters.VOICE, audio_message_handler))

    # Run bot
    application.run_polling(1.0)


if __name__ == '__main__':
    global classifier
    global label_decoder
    global name_mapping
    classifier = build_classifier_speechbrain()
    label_decoder = build_label_decoder()
    name_mapping = build_name_decoder()

    config = read_config(filename = 'config.yaml')
    main(config)







#Убирает из датасета не существующие ссылки
def clear_data(start = "id10001"):
    """
    start - id с которого начать чистку
    """
    path_txt = "D:/speech_project/hse_project/txt"
    url = "https://www.youtube.com/watch?v="
    a = listdir(path_txt)
    deleted = 1
    start_flag = 0
    for person_id in a:

        if (person_id == start):
            start_flag= True
        if(start_flag):
            vid_ids = listdir(path_txt + "/" + person_id)
            for video_id in vid_ids:
                response = requests.get(url  + str(video_id))
                if (response.status_code == 404):
                    os.remove(video_id)
                    deleted += 1
                if(deleted % 100 == 0):
                    print("deledeted = ",deleted)
    print("program ended")
