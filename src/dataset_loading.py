import pandas as pd
import sys
from util.database import connect_database, save_dataset


def process_csv(filename):
    df = pd.read_csv(filename)
    connection = connect_database()
    save_dataset(df, connection, "dataset_raw", True)
    connection.close()


if __name__ == "__main__":
    process_csv(sys.argv[1])