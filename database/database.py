import psycopg2
import uuid
import pandas as pd

def connect_bd(dbname, user, password, host, port):
    db = psycopg2.connect(dbname=dbname,
                          user=user,
                          password=password,
                          host=host,
                          port=port)

    cursor = db.cursor()

    return db, cursor


def get_user_info(db, id):
    query = "SELECT * FROM users"

    df = pd.read_sql_query(query, db)

    df_id = df[df['user_id'] == id]

    info = df_id.to_dict(orient='records')[0]

    return info


def add_user(db, cursor, id, embedding, genero, faixa_etaria, first_date, last_date):

    embedding = embedding.tobytes()
    cursor.execute(
        f"INSERT INTO users VALUES ('{id}', {psycopg2.Binary(embedding)}, {genero},{faixa_etaria}, "
        f"'{first_date}','{last_date}', {'NULL'}, {'NULL'})")
    db.commit()


def remove_user(db, cursor, user_id):
    cursor.execute(f"DELETE FROM users WHERE user_id = %s", (user_id,))
    db.commit()


def add_event(db, cursor, user_id, genero, faixa_etaria, date):
    UUID = str(uuid.uuid1())

    cursor.execute(
        f"INSERT INTO events VALUES ('{UUID}', '{user_id}', {genero},{faixa_etaria}, "
        f"'{date}',{'NULL'}, {'NULL'})")
    db.commit()