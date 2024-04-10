from database.database import *
from time import sleep
db = Database()
db, cursor = db.__connect_bd()
print("Conectado ao banco de dados!")

while True:

    cursor.execute("SELECT DISTINCT user_id FROM events")
    
    qtd_pessoas = len(cursor.fetchall())

    print(f"\nQuantidade de pessoas: {qtd_pessoas}")
    sleep(60)
