import pandas as pd
import numpy as np

def cosine_similarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def get_image_from_camera():
    pass


def process_image(img, mtcnn, resnet, device):
    faces, probs = mtcnn(img, return_prob=True)
    if probs[0] is None:
        return False, None
    indices = []
    for idx, prob in enumerate(probs):
        if prob >= 0.95:
            indices.append(idx)

    else:
        faces_returned = [faces[i] for i in indices]
        return True, faces_returned

# Função para converter a coluna 'embedding' em um array NumPy
def convert_embedding(embedding):
    return np.frombuffer(embedding, dtype=np.float32)
def search_face(embedding, db):
    query = "SELECT user_id, embedding FROM users"

    df = pd.read_sql_query(query, db)
    df['embedding'] = df['embedding'].apply(convert_embedding)
    id_people = dict(zip(df['user_id'], df['embedding']))

    if len(id_people) == 0:
        return 100, None

    scores = []
    ids = []

    for key, value in id_people.items():
        scores.append(cosine_similarity(embedding, value))
        ids.append(key)

    idx = scores.index(min(scores))

    return scores[idx], ids[idx]


def get_face_embedding_rt(img_cropped, resnet, device):
    img_embedding = resnet(img_cropped.unsqueeze(0).to(device))

    embedding = img_embedding.cpu().detach().numpy().reshape(512)

    return embedding
