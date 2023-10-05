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
    mtcnn = MTCNN(device=device, keep_all=True)
    resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
    age_m = Age_Model()
    gender_m = Gender_Model()

    db, cursor = connect_bd('p_counter', 'smartairuser', '7cLjSM0AfIHAVCe', 'smartair-user.coqfqdmu41ep.us-east-2.rds.amazonaws.com', '5432')
    print("Conectado ao banco de dados!")

    v = cv2.VideoCapture(0)
    print("Inicializando câmera!")
    while True:

        ret, img = v.read()
        if not ret:
            break
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #img = get_image_from_camera()
        has_face, faces = process_image(img, mtcnn, resnet, device)

        if not has_face:
            continue

        else:

            for face in faces:

                embedding = get_face_embedding_rt(face, resnet, device)
                score, user_id = search_face(embedding, db)

                if score <= 0.4:

                    user_info = get_user_info(db, user_id)

                    remove_user(db, cursor, user_id)

                    add_user(db, cursor, user_info['user_id'], user_info['embedding'], user_info['gender'],
                             user_info['age_range'],
                             user_info['first_date'], datetime.now())

                    add_event(db, cursor, user_id, user_info['gender'], user_info['age_range'], datetime.now())

                    print("Informação inserida no banco com sucesso!")

                else:

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
