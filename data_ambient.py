from paho.mqtt import client as mqtt_client
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from gender_age_detection import *
import cv2
import time
import datetime
from os import makedirs, path
from utils import *


class DataAmbient:
    
    def __init__(self):
        """
        Classe que representa o script da Data Ambient. Contém todas as funções necessárias para o funcionamento do script.
        """

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # selecionando a GPU

        self.mtcnn = MTCNN(device=self.device) # modelo responsável por extrair as faces de uma imagem
        self.resnet = InceptionResnetV1(pretrained='vggface2', device=self.device).eval() # modelo responsável por gerar o código único representante da face
        self.age_m = Age_Model() # modelo responsável por classificar a idade da pessoa
        self.gender_m = Gender_Model() # modelo responsável por classificar o gênero da pessoa

        self.id_people = dict() # estrutura que armazena o id da pessoa e o seu código representante da face

        self.id_rec = dict() # estrutura que armazena o id da pessoa e quantas vezes ela foi naquele ambiente
        
        self.events = [] # estrutura que armazena os eventos do ambiente, organizada na seguinte ordem: id da pessoa, data do evento, se o evento é um registro, se o evento é uma entrada, se o evento é uma saída, quanto tempo a pessoa permaneceu no ambiente e quanto tempo a pessoa demorou para retornar ao ambiente
        
        self.on_ambient = [] # estrutura que armazena os ids presentes no ambiente no momento
        
        self.possiveis_recorrentes = [] # estrutura que armazena os ids recorrentes que estão no ambiente no momento
        
        self.tempo_permanencia = [] # estrutura que armazena os tempos de permanência das pessoas
        
        self.tempo_recorrencia = [] # estrutura que armazena os tempos de recorrência das pessoas

        self.qtd_recorrentes = 0 # variável que armazena quantas pessoas recorrentes há no ambiente atualmente
        self.gender = None # variável que armazena o gênero da pessoa
        self.age = None # variável que armazena a idade da pessoa

        self.is_empty = True # variável que indica se o ambiente nunca recebeu alguém ou não

        self.total_pessoas = 0 # variável que indica quantas pessoas diferentes já passou no ambiente

        self.TIME_THRESHOLD = 15 # definindo o limiar, em segundos, para detectar uma pessoa novamente

    def extract_face(self, msg):
        """
        Recebe a mensagem MQTT e retorna a face e o nível de confiança da face
        :param msg: mensagem MQTT recebida
        :return: retorna a face e seu nível de confiança
        """
        try:
            self.frame, _ = img_to_numpy(msg) # convertendo a imagem para numpy
            self.face, conf = self.mtcnn(self.frame, return_prob=True) # extraindo a face e sua confiança
            return face, conf
        except:
            _, img = img_to_numpy(msg)  # convertendo a imagem, quando ela vem em outro formato, para numpy
            self.face, conf = self.mtcnn(np.array(img), return_prob=True)  # extraindo a face e sua confiança
            return self.face, conf

    def register_facecode(self):
        """
        Registra o código representante da face da pessoa
        """
        embedding = get_face_embedding_rt(self.face, self.resnet, self.device) # extraindo o código único da face
        self.id_people[self.total_pessoas] = embedding # armazenando o código

    def register_recurrence(self, id):
        """
        Registra a recorrência associada ao id da pessoa
        :param id: id que representa a pessoa
        """
        # tenta ver se o id da pessoa já existe, caso não, registra a primeira visita
        try:
            self.id_rec[id] +=1
        except KeyError:
            self.id_rec[id] = 1

    def save_image(self):
        """
        Função responsável por salvar a imagem da face captada da pessoa
        """

        log_dir = f"./images"
        # cria o diretório caso não exista
        if not path.exists(log_dir):
            makedirs(log_dir)


        # mudando a escala da imagem de BGR para RGB e salvando ela
        img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'images/{self.total_pessoas}.jpg', img)

    def extract_gender_age(self):
        """
        Função responsável por extrair a idade e gênero da face da pessoa
        """

        self.gender = find_gender(self.face.permute(1,2,0).cpu().detach().numpy(), self.gender_m)
        self.age = find_age(self.face.permute(1,2,0).cpu().detach().numpy(), self.age_m)

    def update_event(self, id, data, is_register, is_entry, is_exit, tempo_p, tempo_r):
        """
        Função responsável por registrar e atualizar a estrutura de eventos
        :param id: id que representa a pessoa
        :param data: data em que foi registrado o evento
        :param is_register: variável binária que indica se é um registro ou não
        :param is_entry: variável binária que indica se é uma entrada ou não
        :param is_exit: variável binária que indica se é uma saída ou não
        :param tempo_p: variável que indica o tempo de permanência da pessoa no ambiente
        :param tempo_r: variável que indica o tempo de recorrência da pessoa no ambiente
        """
        self.events.append([id, data, is_register, is_entry, is_exit, tempo_p, tempo_r])

    def update_people_on_ambient(self, is_entry, id):
        """
        Função responsável por atualizar a quantidade de pessoas no ambiente
        :param is_entry: indica se a pessoa está entrando ou saindo do ambiente
        :param id: id que representa a pessoa
        """

        # se a pessoa entrou no ambiente o id dela é adicionado, caso esteja saindo o id é removido
        if is_entry:
            self.on_ambient.append(id)
        else:
            self.on_ambient.remove(id)

    def register_and_entry(self):
        """
        Função responsável por armazenar o registro e a entrada da pessoa
        """

        self.register_facecode() # registrando o id da pessoa e seu código único da face
        self.register_recurrence(self.total_pessoas) # atualizando a lista de pessoas recorrentes e quantas vezes elas visitaram o local
        self.save_image() # salvando a imagem
        self.update_event(self.total_pessoas, datetime.datetime.now(), 1,1,0,np.nan,np.nan) # atualizando a lista de eventos
        self.update_people_on_ambient(1, self.total_pessoas) # atualizando as pessoas que estão no ambiente
        self.extract_gender_age() # extraindo gênero e idade da pessoa
        self.total_pessoas+=1 # atualizando o total de pessoas que já visitaram o ambiente

    def search_faces(self, face):
        """
        Função responsável por procurar a face mais semelhante à face presente
        :return: retorna a lista com os valores de semelhança
        """
        scores = [] # inicializando a lista de valores de semelhança

        # iterando sobre todas as faces e armazenando as distâncias entre cada uma e a face atual
        embedding = get_face_embedding_rt(face, self.resnet, self.device)
        for key, value in self.id_people.items():
            scores.append(cosine_similarity(embedding, value))
        return scores

    def exit_env(self, id, data, tempo_p):
        """
        Função responsável por armazenar a saída da pessoa
        :param id: id que representa a pessoa
        :param tempo_p: tempo que a pessoa permaneceu no ambiente
        :param data: data em que foi registrado o evento
        """
        self.register_recurrence(id) # atualizando a lista de pessoas recorrentes e quantas vezes elas visitaram o local
        self.update_event(id, data, 0,0,1,tempo_p,np.nan) # atualizando a lista de eventos
        self.tempo_permanencia.append(tempo_p) # armazenando o tempo de permanência da pessoa

        # colocando um valor inválido para o gênero e a idade, já que eles foram captados na entrada e não precisam ser armazenados de forma dobrada
        self.gender = -1
        self.age = -1
        self.update_people_on_ambient(0, id) # atualizando as pessoas que estão no ambiente

        # atualizando a lista de possíveis pessoas recorrentes e quantos recorrentes estão no ambiente
        if id in self.possiveis_recorrentes:
            self.qtd_recorrentes -= 1
        else:
            self.possiveis_recorrentes.append(id)

    def entry_env(self, id, data, tempo_r):
        """
        Função responsável por armazenar a entrada da pessoa
        :param id: id que representa a pessoa
        :param tempo_r: tempo que a pessoa demorou para voltar ao ambiente
        :param data: data em que foi registrado o evento
        """
        self.update_event(id, data, 0,1,0,np.nan,tempo_r) # atualizando a lista de eventos
        self.tempo_recorrencia.append(tempo_r) # armazenando o tempo de recorrência da pessoa
        self.extract_gender_age() # extraindo gênero e idade da pessoa
        self.update_people_on_ambient(1, id) # atualizando as pessoas que estão no ambiente

        # atualizando quantos recorrentes estão no ambiente com base na lista de possíveis recorrentes
        if id in self.possiveis_recorrentes:
            self.qtd_recorrentes += 1

    def update_env(self, id):
        """
        Função responsável por atualizar a estrutura de eventos, com base na entrada ou saída da pessoa
        :param id: id que representa a pessoa
        """
        changes = False
        
        # iterando sobre todos os eventos armazenados
        for j in range(len(self.events)-1, -1, -1):

            id_evento = self.events[j][0] # extraindo o id da pessoa no evento que está sendo analisado
            
            if id_evento == id: # confere se o id da pessoa do evento é o mesmo id da pessoa que passou pela câmera
                
                atual = datetime.datetime.now() # extraindo a data atual
                data_evento = self.events[j][1] # extraindo a data do evento

                dif = atual - data_evento # calculando quanto tempo se passou
                tempo = dif.total_seconds() # armazenando o tempo em segundos

                is_entry = self.events[j][3] # analisando se o evento é uma entrada

                # condicional: faz mais de 15 segundos que a pessoa foi detectada e o último evento dela é uma entrada. Então, é uma saída.
                if tempo >= self.TIME_THRESHOLD and is_entry == 1:

                    self.exit_env(id, atual, tempo)
                    changes = True
                    break

                # condicional: faz mais de 15 segundos que a pessoa foi detectada e o último evento dela não é uma entrada. Então, é uma entrada.
                elif tempo >= self.TIME_THRESHOLD and is_entry == 0:

                    self.entry_env(id, atual, tempo)
                    changes = True
                    break
                else:
                    break
                    
            else:
                continue

        return changes


    def register_new_person(self):
        """
        Função responsável por atualizar a estrutura de eventos, com base na entrada ou saída da pessoa
        :param id: id que representa a pessoa
        """
        
        self.register_facecode() # registrando o id da pessoa e seu código único da face
        self.register_recurrence(self.total_pessoas) # atualizando a lista de pessoas recorrentes e quantas vezes elas visitaram o local
        self.save_image() # salvando a imagem
        self.update_event(self.total_pessoas, datetime.datetime.now(), 1,1,0,np.nan,np.nan) # atualizando a lista de eventos
        self.update_people_on_ambient(1, self.total_pessoas) # atualizando as pessoas que estão no ambiente
        self.total_pessoas+=1 # atualizando o total de pessoas que já visitaram o ambiente


    def get_data(self):

        """
        Função responsável por retornar os dados que serão publicados no dashboard, que são, respectivamente:
        -> Quantidade de pessoas no ambiente
        -> Tempo médio de permanência
        -> Tempo médio de recorrência
        -> Quantidade de pessoas recorrentes no ambiente
        -> Gênero da última pessoa que entrou no ambiente
        -> Idade da última pessoa que entrou no ambiente
        """
        return len(self.on_ambient), np.mean(self.tempo_permanencia), np.mean(self.tempo_recorrencia), self.qtd_recorrentes, self.gender, self.age