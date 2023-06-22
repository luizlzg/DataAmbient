from paho.mqtt import client as mqtt_client
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from numpy.linalg import norm
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import time
import datetime
import pandas as pd
from gender_age_detection import *


def get_face_embedding_rt(img_cropped,resnet,device):
    #img_cropped = torch.tensor(img_cropped).permute(2,1, 0).float().to(device)
    img_embedding = resnet(img_cropped.unsqueeze(0).to(device))
    
    embedding = img_embedding.cpu().detach().numpy().reshape(512)

    return embedding


def cosine_similarity(source_representation,test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
    #return np.dot(A,B)/(norm(A)*norm(B))

def multi_scale_retinex(img, sigma_list):
    img = img.astype(np.float32) / 255.0
    img_retinex = np.zeros_like(img)
    for sigma in sigma_list:
        img_blur = cv2.GaussianBlur(img, (0, 0), sigma)
        img_retinex += np.log10(img + 1e-6) - np.log10(img_blur + 1e-6)
    img_retinex = np.power(10, img_retinex / len(sigma_list))
    img_retinex = np.clip(img_retinex, 0, 1)
    img_retinex = (img_retinex * 255).astype(np.uint8)
    img_retinex = cv2.cvtColor(img_retinex, cv2.COLOR_BGR2RGB)
    return img_retinex


def img_to_numpy(mensagem):
    base64_decoded = base64.b64decode(mensagem.payload+ b'==')
    img = Image.open(io.BytesIO(base64_decoded))
    frame = np.array(img)
    return frame,img

def find_age(np_img, agemodel):
    current_img = np.expand_dims(np_img, axis=0)
    pred = np.argmax(agemodel.predict(current_img,verbose=0))
    age_labels = ["Jovem", "Adulto","Idoso"]
    age = age_labels[pred]
    return age

def find_gender(np_img, gender_model):
    current_img = cv2.resize(np_img, (224,224))
    current_img = np.expand_dims(current_img, axis=0)
    gender_predictions = gender_model.predict(current_img, verbose=0)[0,:]
    
    gender_labels = ["Feminino", "Masculino"]
    gender = gender_labels[np.argmax(gender_predictions)]
    
    return gender