import time  # to simulate a real time data, time loop
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
from paho.mqtt import client as mqtt_client
from streamlit_extras.metric_cards import style_metric_cards
import datetime
import json

broker = 'broker.emqx.io'
port = 1883
topic = "data_ambient/iot/device_image"
topic2 = "iot/dashboard"
client_id = f'python-mqtt-luizg77'
username = 'emqx'
password = 'public'

qtd_anterior = 0.
tempo_p_anterior = 0.
tempo_r_anterior = 0.
tempo_p_anterior2 = 0.
tempo_r_anterior2 = 0.
recorrentes_anterior = 0.
taxa_anterior = 0.

pessoas_mes = 0.
recorrentes_mes = 0.
data_mes = datetime.datetime.now()

datas = []
datas2 = []
pessoas_total = []
recorrentes_total = []

ages = []
qtd_age = []
generos = []
qtd_genero = []

st.set_page_config(page_title="Data Ambient Dashboard",page_icon="✅",layout="wide")

st.title("Data Ambient Dashboard")

placeholder = st.empty()


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

def subscribe(client: mqtt_client):

    def on_message(client, userdata, msg):

        global qtd_anterior
        global qtd_genero
        global generos
        global tempo_p_anterior
        global tempo_r_anterior
        global tempo_p_anterior2
        global tempo_r_anterior2
        global recorrentes_anterior
        global taxa_anterior
        global pessoas_mes
        global recorrentes_mes
        
        agr = datetime.datetime.now()
        diff_date = agr - data_mes
        if diff_date.days >= 30:
            recorrentes_mes = 0
            pessoas_mes = 0

        dados = 0
        try:
            dados = json.loads(msg.payload.decode())
        except json.decoder.JSONDecodeError:
            print("Mensagem inválida")

        qtd_pessoas = dados['qtd_pessoas']
        tempo_p = dados['tempo_p']
        tempo_r = dados['tempo_r']
        recorrentes = dados['recorrentes']
        gender = dados['genero']
        age = dados['idade']

        days = {
            0: f"Segunda ({agr.day}/{agr.month})",
            1: f"Terça ({agr.day}/{agr.month})",
            2: f"Quarta ({agr.day}/{agr.month})",
            3: f"Quinta ({agr.day}/{agr.month})",
            4: f"Sexta ({agr.day}/{agr.month})",
            5: f"Sábado ({agr.day}/{agr.month})",
            6: f"Domingo ({agr.day}/{agr.month})"
        }

        datas.append(agr)
        datas2.append(days[agr.weekday()])
        if gender != -1:
            generos.append(gender)
            qtd_genero.append(1)
        else:
            generos.append("Masculino")
            qtd_genero.append(0)
            
        if age != -1:
            ages.append(age)
            qtd_age.append(1)
        else:
            ages.append("Adulto")
            qtd_age.append(0)
            
        pessoas_total.append(qtd_pessoas)
        recorrentes_total.append(recorrentes)

        if tempo_p == tempo_p_anterior:
            tempo_p_anterior = tempo_p_anterior2

        if tempo_r == tempo_r_anterior:
            tempo_r_anterior = tempo_r_anterior2

        if qtd_pessoas > qtd_anterior:
            pessoas_mes += qtd_pessoas - qtd_anterior
        if recorrentes > recorrentes_anterior:
            recorrentes_mes += recorrentes - recorrentes_anterior

        taxa = 0
        if pessoas_mes != 0:
            taxa = round(recorrentes_mes/pessoas_mes, 2)
            
        with placeholder.container():

                # create three columns
                kpi1, kpi2, kpi3 = st.columns(3)

                # fill in those three columns with respective metrics or KPIs
                kpi1.metric(
                    label="Pessoas no ambiente",
                    value=qtd_pessoas,
                    delta=qtd_pessoas - qtd_anterior,
                )

                kpi2.metric(
                    label="Pessoas recorrentes",
                    value=recorrentes,
                    delta=recorrentes - recorrentes_anterior,
                )

                kpi3.metric(
                    label="Tempo médio de permanência",
                    value=f"{round(tempo_p,2)} segundos",
                    delta=f"{round(tempo_p - tempo_p_anterior, 2)} segundos",
                )

                kpi3.metric(
                    label="Tempo médio de recorrência",
                    value=f"{round(tempo_r,2)} segundos",
                    delta=f"{round(tempo_r - tempo_r_anterior,2)} segundos",
                    delta_color = 'inverse',
                )
                kpi3.metric(
                    label="Taxa de recorrência",
                    value=f"{taxa*100} %",
                    delta=f"{(taxa-taxa_anterior)*100} %",
                )

                qtd_anterior = qtd_pessoas
                recorrentes_anterior = recorrentes
                tempo_r_anterior2 = tempo_r_anterior
                tempo_r_anterior = tempo_r
                tempo_p_anterior2 = tempo_p_anterior
                tempo_p_anterior = tempo_p
                taxa_anterior = taxa

                df = pd.DataFrame(dict(tempo=datas, total_de_pessoas=pessoas_total, pessoas_recorrentes=recorrentes_total))
                df2 = pd.DataFrame(dict(tempo=datas2, pessoas=qtd_genero, genero=generos))
                df3 = pd.DataFrame(dict(tempo=datas2, pessoas=qtd_age, faixa_etaria=ages))
                fig = px.line(df, x='tempo', y='total_de_pessoas',
                              title='Variação da quantidade de pessoas por horário', markers=True)

                fig2 = px.histogram(df2, x="tempo", y="pessoas", color='genero', barmode='group',
                                    title='Variação semanal de pessoas por gênero',color_discrete_map={'Masculino': 'blue','Feminino': 'Red'})
                fig3 = px.histogram(df3, x="tempo", y="pessoas",color='faixa_etaria', barmode='group',
                                    title = 'Variação semanal de pessoas por faixa etária',
                                    color_discrete_map={'Jovem': 'blue','Adulto': 'Red','Idoso': 'green'})
                
                kpi1.write(fig)
                kpi1.write(fig2)
                kpi1.write(fig3)
                style_metric_cards()




    client.subscribe(topic2)
    client.on_message = on_message

def run():
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()


if __name__ == '__main__':
    run()
