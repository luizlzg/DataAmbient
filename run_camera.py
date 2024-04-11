import cv2
import time
import os
from main import Camera

# Configura o endereço e a porta do servidor de destino
host = 'localhost'  # Substitua pelo IP público ou nome de domínio do computador de destino
port = 10629

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((host, port))


camera = Camera()

cap = camera.connect_ip()
#cap = camera.connect_local() -> Descomente essa linha e comente a linha acima para obter as imagens da Webcam ao invés de uma câmera externa.

# Define o fator de frame skipping (por exemplo, 2 significa pular a cada 2º quadro)
frame_skip_factor = 1
frame_count = 0

while True:
    # Lê um quadro da captura de vídeo
    ret, frame = cap.read()

    # Verifica se a leitura foi bem-sucedida
    if not ret:
        break

    # Incrementa o contador de quadros
    frame_count += 1

    # Pula frames até que o contador atinja o fator de frame skipping
    if frame_count % frame_skip_factor == 0:
        # Exibe o quadro atual
        cv2.imshow('Webcam', frame)

        data = pickle.dumps(frame)
        message = struct.pack("Q",len(data))+data
        
        client_socket.sendall(message)


    # Verifica se o usuário pressionou a tecla 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura de vídeo e fecha a janela
cap.release()
client_socket.close()
cv2.destroyAllWindows()
