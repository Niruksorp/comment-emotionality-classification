import os
import psycopg2
import pandas as pd
from psycopg2 import sql


def connect_database():
    username = os.getenv('DB_USERNAME')
    password = os.getenv('DB_PASSWORD')
    conn = psycopg2.connect(
        host="postgres://admangarakov:zGKsv6rsX01790iaEivPCxc6TeC6cRvj@dpg-chn1bvak728vrd9a9ed0-a.oregon-postgres"
             ".render.com",
        database="aais",
        user=username,
        password=password)
    return conn


def save_dataset(df, connection, table_name, create_indices=False):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("DROP TABLE IF EXISTS {table}").format(table=sql.Identifier(table_name)))
    if create_indices:
        cursor.execute(sql.SQL("CREATE TABLE {table}(id int primary key, resume text, category text)")
                       .format(table=sql.Identifier(table_name)))
    else:
        cursor.execute(sql.SQL("CREATE TABLE {table}(id int primary key, resume text, category int)")
                       .format(table=sql.Identifier(table_name)))
    for index, row in df.iterrows():
        cursor.execute(sql.SQL("INSERT INTO {table}(id, resume, category) values (%s, %s, %s)")
                       .format(table=sql.Identifier(table_name)),
                       (index if create_indices else row["Id"], row["Resume"], row["Category"]))
    cursor.close()
    connection.commit()
