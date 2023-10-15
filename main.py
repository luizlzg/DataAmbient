from datetime import datetime
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from models.age_gender import *
from database.database import *
from processing.image_process import *
import warnings
import os
warnings.filterwarnings("ignore")

if not os.path.exists('./weights'):
        print(f"O diretório 'weights' não existe. Criando...")
        try:
            # criando o diretório de logs
            os.makedirs('./weights')
        except Exception as e:
            print(f"Não foi possível criar o diretório 'weights'. Erro: {e}")
            exit(1)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # declarando os modelos a serem utilizados
    mtcnn = MTCNN(device=device, keep_all=True)
    resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
    age_m = Age_Model()
    gender_m = Gender_Model()

    try:
        # declarando o objeto do banco de dados a ser utilizado
        db, cursor = connect_bd('p_counter', 'smartairuser', '7cLjSM0AfIHAVCe',
                                'smartair-user.coqfqdmu41ep.us-east-2.rds.amazonaws.com', '5432')
        print("Conectado ao banco de dados!")

    except Exception as eror:
        print(f"Não foi possível conectar ao banco de dados. Erro: {eror}")
        exit(1)

    v = cv2.VideoCapture(0)
    print("Inicializando câmera!")

    while True:
        # capturando as imagens da câmera
        ret, img = v.read()
        if not ret:
            break
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #img = get_image_from_camera()
        # verificando se há faces na imagem
        has_face, faces = process_image(img, mtcnn, resnet, device)

        # se não houver nenhuma face, continua para a próxima imagem
        if not has_face:
            continue

        # se tiver faces, itera sobre elas e...
        else:
            for face in faces:
                # ...obtém o embedding que representa aquela face
                embedding = get_face_embedding_rt(face, resnet, device)
                # faz uma busca semântica no banco de dados a partir do embedding da face detectada
                score, user_id = search_face(embedding, db)

                # se encontrar uma face semelhante, apenas atualiza o horario, mantendo genero e idade
                # TODO: melhorar esse valor arbitrário (colocar um modelo de classificação)
                if score <= 0.4:
                    # TODO: pode ser importante extrair novamente os dados de gênero e faixa etária para garantir mais precisão
                    # buscando as informações do usuário no banco de dados
                    user_info = get_user_info(db, user_id)

                    # atualiza o evento do usuário no banco de dados, removendo o evento anterior e adicionando um novo
                    # TODO: trocar isso por uma cláusula UPDATE
                    remove_user(db, cursor, user_id)
                    add_user(db, cursor, user_info['user_id'], user_info['embedding'], user_info['gender'],
                             user_info['age_range'],
                             user_info['first_date'], datetime.now())
                    add_event(db, cursor, user_id, user_info['gender'], user_info['age_range'], datetime.now())
                    print("Informação inserida no banco com sucesso!")

                else:
                    # se não encontrar uma face semelhante, adiciona uma nova face no banco de dados e extrai as info
                    id = str(uuid.uuid1())
                    genero = find_gender(face, gender_m)
                    faixa_etaria = find_age(face, age_m)
                    add_user(db, cursor, id, embedding, genero,
                             faixa_etaria,
                             datetime.now(), datetime.now())
                    add_event(db, cursor, id, genero, faixa_etaria, datetime.now())
                    print("Informação inserida no banco com sucesso!")


if __name__ == '__main__':
    main()
