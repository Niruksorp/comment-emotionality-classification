from util.database import connect_database, create_data_structure

if __name__ == "__main__":
    connection = connect_database()
    create_data_structure(connection)
    connection.close()
