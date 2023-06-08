import pandas as pd
import psycopg2
from psycopg2 import sql


def connect_database():
    user = 'admangarakov'
    pwd = 'zGKsv6rsX01790iaEivPCxc6TeC6cRvj'
    dbname = 'aais'
    conn = psycopg2.connect(f"postgres://{user}:{pwd}@dpg-chn1bvak728vrd9a9ed0-a.oregon-postgres.render.com/{dbname}")
    return conn


def create_data_structure(connection):
    cursor = connection.cursor()

    cursor.execute(sql.SQL(
        "CREATE TABLE IF NOT EXISTS {table}(id int primary key, comment_message text, emotional_grade integer, version integer)")
                   .format(table=sql.Identifier("dataset")))

    cursor.execute(sql.SQL(
        "CREATE TABLE IF NOT EXISTS {table}(model_id int primary key, model_name text, weights bytea, version integer, score decimal, data_version integer)")
                   .format(table=sql.Identifier("model")))

    cursor.execute(sql.SQL(
        "CREATE TABLE IF NOT EXISTS {table}(id int primary key, comment_message text, emotional_grade integer)")
                   .format(table=sql.Identifier("preprocessed_dataset")))

    cursor.execute(sql.SQL(
        "CREATE TABLE IF NOT EXISTS {table}(id int primary key, model_id integer, model_name text)")
                   .format(table=sql.Identifier("deploy")))

    cursor.execute(sql.SQL(
        "CREATE TABLE IF NOT EXISTS {table}(lock integer)")
                   .format(table=sql.Identifier("deploy_lock")))
    cursor.close()
    connection.commit()


def save_dataset(df, connection):
    cursor = connection.cursor()
    cursor.execute('DROP TABLE IF EXISTS dataset')
    cursor.execute(sql.SQL(
        "CREATE TABLE IF NOT EXISTS {table}(id int primary key, comment_message text, emotional_grade integer, version integer)")
                   .format(table=sql.Identifier("dataset")))
    values = []
    for index, row in df.iterrows():
        values.append((index, row["Comment"], row["Sentiment"], "1"))

    args = ','.join(cursor.mogrify("(%s,%s,%s,%s)", i).decode('utf-8')
                    for i in values)

    cursor.execute('INSERT INTO dataset (id, comment_message, emotional_grade, version) VALUES' + (args))

    cursor.close()
    connection.commit()


def clear_preprocessed_data():
    connection = connect_database()
    cursor = connection.cursor()
    cursor.execute('DROP TABLE IF EXISTS preprocessed_dataset')
    cursor.execute(sql.SQL(
        "CREATE TABLE IF NOT EXISTS {table}(id int primary key, comment_message text, emotional_grade integer)")
                   .format(table=sql.Identifier("preprocessed_dataset")))
    cursor.close()
    connection.commit()
    connection.close()


def get_start_ind_for_prepared_ds():
    connection = connect_database()
    cursor = connection.cursor()
    cursor.execute('SELECT id FROM preprocessed_dataset order by id desc limit 1')
    max_id = cursor.fetchall()
    if len(max_id) == 0:
        max_id = 0
    else:
        max_id = int(max_id[0][0])
    cursor.close()
    connection.commit()
    connection.close()
    return max_id


def save_prepared_dataset(df, start_ind):
    connection = connect_database()
    cursor = connection.cursor()
    values = []
    for index, row in df.iterrows():
        values.append((index + start_ind, row["CommentMessage"], row["Sentiment"]))

    args = ','.join(cursor.mogrify("(%s,%s,%s)", i).decode('utf-8')
                    for i in values)

    cursor.execute('INSERT INTO preprocessed_dataset (id, comment_message, emotional_grade) VALUES' + args)
    cursor.close()
    connection.commit()
    connection.close()


def download_dataset(table_name):
    connection = connect_database()
    cursor = connection.cursor()
    cursor.execute(sql.SQL("SELECT * FROM {table} order by id ").format(table=sql.Identifier(table_name)))
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
    connection.commit()
    connection.close()
    return df


def save_model(model, name, score, time):
    insert_model_sql = '''
    INSERT INTO model (model_id, model_name, weights, version, score, time, final_score, data_version) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    '''
    connection = connect_database()
    cursor = connection.cursor()
    cursor.execute(sql.SQL("SELECT MAX(version) from model group by model_name"))
    model_version = cursor.fetchone()
    if model_version is None or len(model_version) == 0:
        model_version = 1
    else:
        model_version = int(model_version[0]) + 1

    cursor.execute(sql.SQL("SELECT model_id from model order by model_id desc limit 1"))
    model_id = cursor.fetchone()
    if model_id is None or len(model_id) == 0:
        model_id = 1
    else:
        model_id = int(model_id[0]) + 1

    data_version = 1
    final = score * 10 - float(time) / 10
    cursor.execute(insert_model_sql,
                   (model_id, name, psycopg2.Binary(model), model_version, score, time, final, data_version))
    connection.commit()
    connection.close()


def load_latest_model_by_name(name):
    connection = connect_database()
    cursor = connection.cursor()
    cursor.execute("SELECT MAX(version) from model where model_name = %s", (name,))
    version = cursor.fetchone()
    if version is not None and len(version) > 0:
        version = int(version[0])
    else:
        version = 1
    select_latest_model_by_name_query = '''
        SELECT weights 
        FROM model
        WHERE model_name = %s
        and version = %s
    '''
    cursor.execute(select_latest_model_by_name_query, (name, version,))
    model_memview = cursor.fetchone()[0]
    if model_memview is None:
        raise Exception(f'Model not found by name: {name}!')
    return bytes(model_memview)


def best_model_deploy():
    connection = connect_database()
    cursor = connection.cursor()
    cursor.execute(
        sql.SQL("SELECT model_id, model_name FROM {table} WHERE final_score = ( SELECT MAX(final_score) FROM {table} )")
        .format(table=sql.Identifier("model")))
    result = cursor.fetchall()

    cursor.execute(sql.SQL("SELECT MAX(id) from deploy"))
    id = cursor.fetchone()
    if id is None or len(id) == 0:
        id = 1
    else:
        id = int(id[0]) + 1

    cursor.execute('INSERT INTO deploy (id, model_id, model_name) VALUES(%s, %s, %s)',
                   (id, result[0][0], result[0][1]))
    cursor.close()
    connection.commit()
    connection.close()


def best_model():
    connection = connect_database()
    cursor = connection.cursor()
    cursor.execute(sql.SQL("SELECT model_id, model_name FROM {table} ORDER BY id DESC LIMIT 1")
                   .format(table=sql.Identifier("deploy")))
    result = cursor.fetchall()
    cursor.execute(sql.SQL('SELECT weights FROM model WHERE model_id = %s'), (result[0][0],))

    data = cursor.fetchone()[0]
    cursor.close()
    connection.commit()
    connection.close()

    return bytes(data), result[0][1]


def get_modell(name):
    connection = connect_database()
    cursor = connection.cursor()

    cursor.execute(
        sql.SQL('SELECT weights FROM model WHERE model_name = %s ORDER BY score DESC LIMIT 1'), (name,))

    data = cursor.fetchone()[0]
    cursor.close()
    connection.commit()
    connection.close()

    return bytes(data), name
