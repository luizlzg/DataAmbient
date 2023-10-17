## raspberry

# Importing Libraries
import cv2
import paho.mqtt.client as mqtt
import base64
import os

# Raspberry PI IP address
MQTT_BROKER = "broker.emqx.io"
# Topic on which frame will be published
MQTT_SEND = "data_ambient/iot/hyundai_prototype/device_image"

# Paho-MQTT Clinet
client = mqtt.Client()
# Establishing Connection with the Broker
client.connect(MQTT_BROKER)

camera_ip = "192.168.100.198"  # Substitua pelo endereço IP da sua câmera
porta_rtsp = "554"  # Porta RTSP padrão
usuario = "admin"  # Substitua pelo nome de usuário da câmera
senha = "DataAmbient777"  # Substitua pela senha da câmera

RTSP_URL = f'rtsp://{usuario}:{senha}@{camera_ip}:{porta_rtsp}/cam/realmonitor?channel=1&subtype=0'

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

v = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

while True:
    # Read Frame
    _, frame = v.read()
    # Encoding the Frame
    _, buffer = cv2.imencode('.jpg', frame)
    # Converting into encoded bytes
    jpg_as_text = base64.b64encode(buffer)
    # Publishig the Frame on the Topic home/server
    client.publish(MQTT_SEND, jpg_as_text)
