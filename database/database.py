import psycopg2
import uuid
import pandas as pd
from processing.image_process import cosine_similarity, convert_embedding


class Database:
    # Inicialização do banco de dados com as variáveis necessárias para fazer a conexão à ele.
    def __init__(self, db_name='p_counter', user='smartairuser', password='7cLjSM0AfIHAVCe',
                 host='smartair-user.coqfqdmu41ep.us-east-2.rds.amazonaws.com', port='5432'):
        self.db_name = db_name
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.db_object = None
        self.cursor = None

    # Método para conectar ao banco de dados.
    def connect(self):
        print("lucas")
        try:
            # declarando o objeto do banco de dados a ser utilizado
            self.db_object, self.cursor = self.__connect_bd()
            print("Conectado ao banco de dados!")
        except Exception as eror:
            print(f"Não foi possível conectar ao banco de dados. Erro: {eror}")
            exit(1)

    # Método que conectar ao banco de dados e retorna o objeto referente à ele, que é necessário para realizar todas as manipulações no banco.
    def __connect_bd(self):
        db = psycopg2.connect(dbname=self.db_name,
                              user=self.user,
                              password=self.password,
                              host=self.host,
                              port=self.port)

        cursor = db.cursor()

        return db, cursor
    
    def get_connection(self):
        return self.__connect_bd()

    def get_user_info(self, id_: str):
        """
        Essa função retorna as informações de um usuário cadastrado no banco de dados.
        :param id_: o id do usuário -> str
        :return: as informações do usuário -> dict
        """
        query = "SELECT * FROM users"
        df = pd.read_sql_query(query, self.db_object)
        df_id = df[df['user_id'] == id_]
        info = df_id.to_dict(orient='records')[0]
        return info

    def add_user(self, id_: str, embedding, genero: int, faixa_etaria: int, first_date, last_date):
        """
        Essa função adiciona um usuário ao banco de dados.
        :param id_: o id do usuário -> str
        :param embedding: o embedding da face do usuário -> np.array
        :param genero: valor inteiro que representa o gênero do usuário -> int
        :param faixa_etaria: valor inteiro que representa a faixa etária do usuário -> int
        :param first_date: timestamp da primeira vez que o usuário foi detectado -> datetime
        :param last_date: timestamp da última vez que o usuário foi detectado -> datetime
        :return: None
        """
        embedding = embedding.tobytes()
        self.cursor.execute(
            f"INSERT INTO users VALUES ('{id_}', {psycopg2.Binary(embedding)}, {genero},{faixa_etaria}, "
            f"'{first_date}','{last_date}', {'NULL'}, {'NULL'})")
        self.db_object.commit()

    def remove_user(self, user_id):
        """
        Essa função remove um usuário do banco de dados.
        :param user_id: o id do usuário -> str
        :return: None
        """
        self.cursor.execute(f"DELETE FROM users WHERE user_id = %s", (user_id,))
        self.db_object.commit()

    def add_event(self, user_id, genero, faixa_etaria, date):
        """
        Essa função adiciona um evento ao banco de dados.
        :param user_id: o id do usuário -> str
        :param genero: o gênero do usuário -> int
        :param faixa_etaria: a faixa etária do usuário -> int
        :param date: timestamp do evento -> datetime
        :return: None
        """
        UUID = str(uuid.uuid1())
        self.cursor.execute(
            f"INSERT INTO events VALUES ('{UUID}', '{user_id}', {genero},{faixa_etaria}, "
            f"'{date}',{'NULL'}, {'NULL'})")
        self.db_object.commit()

    def search_face(self, embedding):
        """
        Essa função faz uma busca semântica no banco de dados, comparando o embedding da face detectada com os embeddings
        de todas as faces cadastradas no banco de dados. A face com menor distância é a que mais se assemelha à face
        detectada.
        :param embedding: o embedding da face detectada -> np.array
        :return: a distância entre o embedding da face detectada e o id da face mais semelhante no banco de dados
        """
        query = "SELECT user_id, embedding FROM users"

        df = pd.read_sql_query(query, self.db_object)
        df['embedding'] = df['embedding'].apply(convert_embedding)
        id_people = dict(zip(df['user_id'], df['embedding']))

        if len(id_people) == 0:
            # TODO: entender o que é isso aqui
            return 100, None

        scores = []
        ids = []

        for key, value in id_people.items():
            scores.append(cosine_similarity(embedding, value))
            ids.append(key)

        idx = scores.index(min(scores))

        return scores[idx], ids[idx]
