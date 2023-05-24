import os
import psycopg2
import pandas as pd
from psycopg2 import sql


def connect_database():
    conn = psycopg2.connect("postgres://admangarakov:zGKsv6rsX01790iaEivPCxc6TeC6cRvj@dpg-chn1bvak728vrd9a9ed0-a.oregon-postgres.render.com/aais")
    return conn


def create_data_structure(connection):
    cursor = connection.cursor()

    cursor.execute(sql.SQL("CREATE TABLE {table}(id int primary key, comment_message text, emotional_grade integer, version integer)")
                   .format(table=sql.Identifier("Dataset")))

    cursor.execute(sql.SQL(
        "CREATE TABLE {table}(model_id int primary key, model_name text, weights text, version integer, f1_score decimal, data_version integer)")
                   .format(table=sql.Identifier("Model")))

    cursor.execute(sql.SQL(
        "CREATE TABLE {table}(id int primary key, commemt_message text, emmotional_grade integer)")
                   .format(table=sql.Identifier("Preprocessed_Dataset")))
    cursor.close()
    connection.commit()


def save_dataset(df, connection):
    cursor = connection.cursor()

    for index, row in df.iterrows():
        cursor.execute(sql.SQL("INSERT INTO {table}(id, comment_message, emotional_grade, version) values (%s, %s, %s, %s)")
                       .format(table=sql.Identifier("Dataset")),
                       (row["Unnamed: 0"], row["Comment"], row["Sentiment"]), "1")
    cursor.close()
    connection.commit()
