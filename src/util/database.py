import os
import psycopg2
import pandas as pd
from psycopg2 import sql


def connect_database():
    user = 'admangarakov'
    pwd = os.getenv('DB_PASS')
    dbname = 'aais'
    conn = psycopg2.connect(f"postgres://{user}:{pwd}@dpg-chn1bvak728vrd9a9ed0-a.oregon-postgres.render.com/{dbname}")
    return conn


def create_data_structure(connection):
    cursor = connection.cursor()

    cursor.execute(sql.SQL(
        "CREATE TABLE {table}(id int primary key, comment_message text, emotional_grade integer, version integer)")
                   .format(table=sql.Identifier("dataset")))

    cursor.execute(sql.SQL(
        "CREATE TABLE {table}(model_id int primary key, model_name text, weights text, version integer, f1_score decimal, data_version integer)")
                   .format(table=sql.Identifier("model")))

    cursor.execute(sql.SQL(
        "CREATE TABLE {table}(id int primary key, comment_message text, emotional_grade integer)")
                   .format(table=sql.Identifier("preprocessed_dataset")))
    cursor.close()
    connection.commit()


def save_dataset(df, connection):
    cursor = connection.cursor()

    values = []
    for index, row in df.iterrows():
        values.append((index, row["Comment"], row["Sentiment"], "1"))

    args = ','.join(cursor.mogrify("(%s,%s,%s,%s)", i).decode('utf-8')
                    for i in values)

    cursor.execute('INSERT INTO dataset (id, comment_message, emotional_grade, version) VALUES' + (args))

    cursor.close()
    connection.commit()


def save_prepared_dataset(df, connection):
    cursor = connection.cursor()
    cursor.execute('DROP TABLE IF EXISTS preprocessed_dataset')
    cursor.execute(sql.SQL(
        "CREATE TABLE {table}(id int primary key, comment_message text, emotional_grade integer)")
                   .format(table=sql.Identifier("preprocessed_dataset")))
    values = []
    for index, row in df.iterrows():
        values.append((index, row["CommentMessage"], row["Sentiment"]))

    args = ','.join(cursor.mogrify("(%s,%s,%s)", i).decode('utf-8')
                    for i in values)

    cursor.execute('INSERT INTO preprocessed_dataset (id, comment_message, emotional_grade) VALUES' + (args))

    cursor.close()
    connection.commit()


def download_dataset(connection, table_name):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("SELECT * FROM {table}").format(table=sql.Identifier(table_name)))
    result = cursor.fetchall()
    comment_message = []
    emotional_grade = []
    ids = []
    for row in result:
        ids.append(row[0])
        comment_message.append(row[1])
        emotional_grade.append(row[2])
    df = pd.DataFrame({"CommentMessage": comment_message, "Sentiment": emotional_grade})
    cursor.close()
    return df
