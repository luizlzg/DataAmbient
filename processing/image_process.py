import pandas as pd
import numpy as np

# Função que calcula a distância cosseno. Ela é utilizada para a calcular a distância entre 2 faces, de modo que quanto mais próximo de 0, mais as faces se parecem.
def cosine_similarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

# Função responsável por processar a imagem proveniente da câmera, detectando e retornando as faces presentes. Caso não haja nenhuma face, é retornado False.
def process_image(img, mtcnn, resnet):
    faces, probs = mtcnn(img, return_prob=True) # detectando faces na imagem/frame
    if probs[0] is None:
        return False, None
    indices = []
    for idx, prob in enumerate(probs):
        if prob >= 0.95:
            indices.append(idx)

    else:
        faces_returned = [faces[i] for i in indices]
        return True, faces_returned


# Função para converter a coluna 'embedding' (armazena o embedding no banco de dados) em um array NumPy
def convert_embedding(embedding):
    """
    Essa função converte o embedding, que está no formato de bytes, para um array NumPy.
    :param embedding: bytes
    :return: np.array
    """
    return np.frombuffer(embedding, dtype=np.float32)


# Função que obtém o embedding da face.
def get_face_embedding_rt(img_cropped, resnet, device):
    img_embedding = resnet(img_cropped.unsqueeze(0).to(device))

    embedding = img_embedding.cpu().detach().numpy().reshape(512)

    return embedding
