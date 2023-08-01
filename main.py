from paho.mqtt import client as mqtt_client
import numpy as np
import json
from data_ambient import *

# definindo as configurações do MQTT

broker = 'broker.emqx.io' # endereço do broker
port = 1883 # porta do broker
topic = "data_ambient/iot/device_image" # 1º tópico: responsável por captar as imagens da webcam
topic2 = "iot/dashboard" # 2º tópico: responsável por enviar os dados para o dashboard
client_id = f'python-mqtt-luizg' # pode colocar qualquer coisa
username = 'emqx' # usuário do broker
password = 'public' # senha do broker


# definindo o objeto DataAmbient, responsável por registrar as movimentações no ambiente
dta = DataAmbient()


# função que conecta o dispositivo ao broker MQTT
def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
            client.subscribe(topic)
        else:
            print("Failed to connect, return code %d\n", rc)

    # Conectando o dispositivo:
    client = mqtt_client.Client(client_id)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

# função responsável por publicar os dados captados pelo objeto DataAmbient. Esses dados publicados servirão para a criação do dashboard
def publish(client):

    qtd_pessoas, tempo_p, tempo_r, qtd_recorrentes, gender, age = dta.get_data() # obtendo os dados do objeto DataAmbient, com base no que ele registrou

    # passando os dados para uma estrutura de dicionário
    dados = {'qtd_pessoas': qtd_pessoas, 'tempo_p': tempo_p,
             'tempo_r': tempo_r, 'recorrentes': qtd_recorrentes, 'genero': gender, 'idade': age}

    
    json_data = json.dumps(dados) # criando um json dos dados
    result = client.publish(topic2, json_data) # publicando os dados

    # analisando se os dados foram publicados com sucesso
    status = result[0]
    if status != 0:
        print(f"Failed to send message to topic {topic2}")

# função que realiza a inscrição do dispositivo para receber as imagens da webcam
def subscribe(client: mqtt_client):
    # função que faz o processamento da imagem recebida
    def on_message(client, userdata, msg):


        face, conf = dta.extract_face(msg) # extraindo a face e seu nível de confiança da imagem

        print(f"Conf:{conf}") # imprimindo a confiança na tela

        # verificando se a confiança não é do tipo "None". Caso seja, é porque nenhuma face foi detectada
        if isinstance(conf, type(None)):
            conf = 0

        # verificando se há mais de 95% de confiança na detecção da face. Caso não tenha, a imagem é descartada
        if conf < 0.95:
            return
            
        # verificando se o objeto Data Ambient está vazio, ou seja, se ninguém nunca visitou aquele ambiente:
        if dta.is_empty:
            if len(face) != 0:
                dta.register_and_entry() # caso ninguem nunca tenha visitado o ambiente, é armazenado o registro da pessoa e sua entrada
                dta.is_empty = False
                publish(client) # publicando os dados no dashboard

        # caso o objeto Data Ambient não esteja vazio, ou seja, se alguma pessoa já visitou o ambiente:
        else:

            scores = np.array(dta.search_faces(face)) # fazendo uma busca por todas as faces armazenadas e retornando uma lista com a distância da face atual para as facas armazenadas
            print(scores) # imprimindo na tela a lista de distâncias
            min_value = np.min(scores) # extraindo a menor distância da lista

            # caso a menor distância seja menor ou igual a 0.4, quer dizer que aquela pessoa já visitou o ambiente:
            if min_value <= 0.4:

                # extraindo a identificação da pessoa que já visitou o ambiente
                ident = np.argmin(scores)
                
                changes = dta.update_env(ident) # atualizando o objeto DataAmbient, analisando se a pessoa está entrando ou saindo e captando as outras informações
                if changes:
                    publish(client) # publicando os dados no dashboard
                return

            # caso a menor distância não seja menor ou igual a 0.4, quer dizer que há uma nova pessoa visitando o ambiente:
            else:
                if len(face) != 0:
                    
                    dta.register_new_person() # registrando uma nova pessoa no objeto DataAmbient
                    publish(client) # publicando os dados no dashboard
                    return

    client.subscribe(topic)
    client.on_message = on_message


def run():
    client = connect_mqtt()
    try:
        subscribe(client)
        client.loop_forever()
    except Exception as e:
        print(e)


if __name__ == '__main__':
    run()
