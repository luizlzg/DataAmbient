from paho.mqtt import client as mqtt_client
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from numpy.linalg import norm
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from gender_age_detection import *
import cv2
import time
import datetime
import pandas as pd
import json
import pickle
from utils import *

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

broker = '34.125.226.239'
port = 1883
topic = "data_ambient/iot/device_image"
topic2 = "iot/dashboard"
client_id = f'python-mqtt-luizg'
username = 'projetoiot'
password = 'projetoiot'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(device=device)
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
age_m = Age_Model()
gender_m = Gender_Model()

id_people = dict()

id_rec = dict()

events = []

qtd_recorrentes = 0

on_ambient = []

possiveis_recorrentes = []

tempo_permanencia = []

tempo_recorrencia = []

gender = None
age = None


def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    # Set Connecting Client ID
    client = mqtt_client.Client(client_id)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def publish(client):
    global qtd_recorrentes
    global gender
    global age

    dados = {'qtd_pessoas': len(on_ambient), 'tempo_p': np.mean(tempo_permanencia),
             'tempo_r': np.mean(tempo_recorrencia), 'recorrentes': qtd_recorrentes, 'genero': gender, 'idade': age}
    json_data = json.dumps(dados)
    result = client.publish(topic2, json_data)
    status = result[0]
    if status != 0:
        print(f"Failed to send message to topic {topic2}")


def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):

        global qtd_recorrentes
        global gender
        global age

        try:
            frame, _ = img_to_numpy(msg)
            face, conf = mtcnn(frame, return_prob=True)
        except:
            _, img = img_to_numpy(msg)
            face, conf = mtcnn(np.array(img), return_prob=True)

        print(f"Conf:{conf}")
        if isinstance(conf, type(None)):
            conf = 0
        if conf >= 0.95:
            scores = []
            if len(face) != 0:
                #img_cropped = cv2.resize(img_cropped1, (224, 224))
                #img_cropped = multi_scale_retinex(img_cropped, [15, 80, 250])
                embedding = get_face_embedding_rt(face, resnet, device)
        else:
            return

        if len(id_people) == 0:
            if len(face) != 0:
                id_people[0] = embedding
                id_rec[0] = 1

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite('images/0.jpg', img)

                events.append([0, datetime.datetime.now(), 1, 1, 0, np.nan, np.nan])
                on_ambient.append(0)
                gender = find_gender(face.permute(1,2,0).cpu().detach().numpy(), gender_m)
                age = find_age(face.permute(1,2,0).cpu().detach().numpy(), age_m)
                publish(client)

        else:
            for key, value in id_people.items():
                scores.append(cosine_similarity(embedding, value))

            scores = np.array(scores)
            print(scores)
            min_value = np.min(scores)
            #max_value = np.max(scores)

            if round(min_value, 2) <= 0.4:

                ident = np.argmin(scores)
                #ident = np.argmax(scores)
                for j in range(len(events), -1, -1):

                    if events[j - 1][0] == ident:

                        atual = datetime.datetime.now()

                        dif = atual - events[j - 1][1]

                        if dif.total_seconds() >= 15 and events[j - 1][3] == 1:

                            events.append([ident, atual, 0, 0, 1, dif.total_seconds(), np.nan])
                            tempo_permanencia.append(dif.total_seconds())
                            gender = -1
                            age = -1
                            on_ambient.remove(ident)
                            try:
                                id_rec[str(ident)] += 1
                            except KeyError:
                                id_rec[ident] += 1

                            if ident in possiveis_recorrentes:
                                qtd_recorrentes -= 1
                            else:
                                possiveis_recorrentes.append(ident)
                            publish(client)

                            return
                        elif dif.total_seconds() >= 15 and events[j - 1][3] == 0:

                            events.append([ident, atual, 0, 1, 0, np.nan, dif.total_seconds()])
                            tempo_recorrencia.append(dif.total_seconds())
                            on_ambient.append(ident)
                            gender = find_gender(face.permute(1,2,0).cpu().detach().numpy(), gender_m)
                            age = find_age(face.permute(1,2,0).cpu().detach().numpy(), age_m)
                            if ident in possiveis_recorrentes:
                                qtd_recorrentes += 1
                            publish(client)
                            return
                        else:
                            return
                    else:
                        continue
            else:
                if len(face) != 0:
                    
                    gender = find_gender(face.permute(1,2,0).cpu().detach().numpy(), gender_m)
                    age = find_age(face.permute(1,2,0).cpu().detach().numpy(), age_m)
                    
                    id_rec[len(id_people)] = 1
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(f'images/{len(id_people)}.jpg', img)
                    id_people[len(id_people)] = embedding

                    events.append([len(id_people) - 1, datetime.datetime.now(), 1, 1, 0, np.nan, np.nan])
                    on_ambient.append(len(id_people) - 1)
                    publish(client)

    client.subscribe(topic)
    client.on_message = on_message


def run():
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()


if __name__ == '__main__':
    run()
