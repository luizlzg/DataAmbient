from datetime import datetime
import socket
import cv2
import pickle
import struct
import psycopg2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
#from models.age_gender import *
from database.database import Database
from processing.image_process import *
import warnings
import os
import uuid

warnings.filterwarnings("ignore")

# --------------------------------------------------------- verificando e criando os diretórios necessários
if not os.path.exists('./weights'):
    print(f"O diretório 'weights' não existe. Criando...")
    try:
        # criando o diretório de logs
        os.makedirs('./weights')
    except Exception as e:
        print(f"Não foi possível criar o diretório 'weights'. Erro: {e}")
        exit(1)

if not os.path.exists('./faces'):
    print(f"O diretório 'faces' não existe. Criando...")
    try:
        # criando o diretório de logs
        os.makedirs('./faces')
    except Exception as e:
        print(f"Não foi possível criar o diretório 'faces'. Erro: {e}")
        exit(1)

if not os.path.exists('./images_camera'):
    print(f"O diretório 'images_camera' não existe. Criando...")
    try:
        # criando o diretório de logs
        os.makedirs('./images_camera')
    except Exception as e:
        print(f"Não foi possível criar o diretório 'images_camera'. Erro: {e}")
        exit(1)


# ---------------------------------------------------------

# --------------------------------------------------------- construindo classe para a captura de imagens e processamento
class Camera:
    def __init__(self, ip_bool=True, camera_ip="192.168.2.34", porta_rtsp="554", usuario="admin",
                 senha="DataAmbient777"):
        if ip_bool:
            self.ip_bool = ip_bool
            # declarando as variáveis de conexão com a câmera
            self.camera_ip = camera_ip  # Substitua pelo endereço IP da sua câmera
            self.porta_rtsp = porta_rtsp  # Porta RTSP padrão
            self.usuario = usuario  # Substitua pelo nome de usuário da câmera
            self.senha = senha  # Substitua pela senha da câmera

            # declarando a URL de conexão com a câmera
            self.RTSP_URL = f'rtsp://{self.usuario}:{self.senha}@{self.camera_ip}:{self.porta_rtsp}/cam/realmonitor?channel=1&subtype=0'
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
        else:
            self.ip_bool = ip_bool

    def connect_ip(self):
        """
        Essa função inicializa a câmera IP e retorna o objeto de captura de vídeo.
        :return: cv2.VideoCapture
        """
        # inicializando a câmera IP
        v = cv2.VideoCapture(self.RTSP_URL, cv2.CAP_FFMPEG)
        print("Câmera inicializada!")
        return v

    def connect_local(self, camera_id=0):
        """
        Essa função inicializa a câmera local e retorna o objeto de captura de vídeo.
        :return: cv2.VideoCapture
        """
        # inicializando a câmera local
        v = cv2.VideoCapture(camera_id)
        print("Câmera inicializada!")
        return v


# ---------------------------------------------------------


# --------------------------------------------------------- construindo classe para a captura de imagens e processamento
class AgeGenderInference:
    def __init__(self, save_image=True):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # declarando os modelos a serem utilizados
        self.mtcnn = MTCNN(device=self.device, keep_all=True)
        self.resnet = InceptionResnetV1(pretrained='vggface2', device=self.device).eval()
        #self.age_m = Age_Model()
        #self.gender_m = Gender_Model()
        # conectando outras classes
        self.camera = Camera()
        self.db = Database()
        self.save_image = save_image
        # inicializando as conexoes
        self.v = self.camera.connect_local()  # TROCAR PARA IP CASO PRECISE
        self.db.connect()

    def capture_video(self):
        # capturando as imagens da câmera
        ret, img = self.v.read()
        if not ret:
            return False, None
        return True, img

    def get_info_from_face(self, face, img):
        # ...obtém o embedding que representa aquela face
        embedding = get_face_embedding_rt(face, self.resnet, self.device)
        # faz uma busca semântica no banco de dados a partir do embedding da face detectada
        score, user_id = self.db.search_face(embedding)

        # se encontrar uma face semelhante, apenas atualiza o horario, mantendo genero e idade
        # TODO: melhorar esse valor arbitrário (colocar um modelo de classificação)
        if score <= 0.4:
            # TODO: pode ser importante extrair novamente os dados de gênero e faixa etária para garantir mais precisão
            # buscando as informações do usuário no banco de dados
            user_info = self.db.get_user_info(user_id)

            # atualiza o evento do usuário no banco de dados, removendo o evento anterior e adicionando um novo
            # TODO: trocar isso por uma cláusula UPDATE
            self.db.remove_user(user_id)
            self.db.add_user(user_info['user_id'], user_info['embedding'], user_info['gender'],
                             user_info['age_range'], user_info['first_date'], datetime.now())
            self.db.add_event(user_id, user_info['gender'], user_info['age_range'], datetime.now())
            print("Informação inserida no banco com sucesso!")

            if self.save_image:
                # salvando imagem e face
                cv2.imwrite(f'images_camera/{user_id}.jpg', img)
                cv2.imwrite(f'faces/{user_id}.jpg',
                            (face.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8))

        else:
            # se não encontrar uma face semelhante, adiciona uma nova face no banco de dados e extrai as info
            id = str(uuid.uuid1())
            genero = -1
            faixa_etaria = -1
            self.db.add_user(id, embedding, genero, faixa_etaria, datetime.now(), datetime.now())
            self.db.add_event(id, genero, faixa_etaria, datetime.now())
            print("Informação inserida no banco com sucesso!")

            if self.save_image:
                # salvando imagem e face
                cv2.imwrite(f'images_camera/{user_id}.jpg', img)
                cv2.imwrite(f'faces/{user_id}.jpg',
                            (face.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8))

    def main(self):
        frame_count = 0

        # Configura o endereço e a porta do servidor
        host = '0.0.0.0'  # Ou o IP público do computador de destino
        port = 10629

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, port))
        server_socket.listen(1)

        print(f"Servidor esperando conexão em {host}:{port}")

        client_socket, client_address = server_socket.accept()
        print(f"Conexão estabelecida com {client_address}")

        data = b""
        payload_size = struct.calcsize("Q")
        while True:
            while len(data) < payload_size:
                data += client_socket.recv(4 * 1024)
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]
            while len(data) < msg_size:
                data += client_socket.recv(4 * 1024)
            frame_data = data[:msg_size]
            data = data[msg_size:]

            img = pickle.loads(frame_data)
            frame_skip_factor = 10
            frame_count += 1

            if frame_count % frame_skip_factor == 0:
                # verificando se há faces na imagem
                has_face, faces = process_image(img, self.mtcnn, self.resnet)

                # se não houver nenhuma face, continua para a próxima imagem
                if not has_face:
                    continue

                # se tiver faces, itera sobre elas e...
                else:
                    for face in faces:
                        try:
                            self.get_info_from_face(face, img)
                        except:
                            self.db.connect()
                            self.get_info_from_face(face, img)


if __name__ == '__main__':
    agi = AgeGenderInference(save_image=False)
    agi.main()
