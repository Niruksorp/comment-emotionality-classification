from util.database import connect_database, save_prepared_dataset, download_dataset

import os
import torch
import time
import numpy as np

from transformers import AutoTokenizer, AutoModel

DATASET_TABLE_NAME = 'dataset'
PREPROCESSED_DATASET = 'preprocessed_dataset'

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
MAX_SEQUENCE_LENGTH = 2048
model.max_seq_length = MAX_SEQUENCE_LENGTH

batch_size = 8


def get_embs(batch):
    tokens = tokenizer(batch, max_length=MAX_SEQUENCE_LENGTH, padding="max_length", truncation=True)["input_ids"]
    tokens = torch.from_numpy(np.array(tokens))
    with torch.no_grad():
        embeddings = model(tokens)
        embeddings = embeddings.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings


def main():
    connection = connect_database()
    dataset = download_dataset(connection, DATASET_TABLE_NAME)
    vectorized_msg = []
    start_time = time.time()
    for start_pos in range(0, len(dataset), batch_size):
        if start_pos + batch_size >= len(dataset):
            break
        batch = dataset.iloc[start_pos:start_pos + batch_size]
        text_data = batch['CommentMessage'].tolist()
        vectorized_msg += get_embs(text_data).numpy().tolist()
        print(f'Processed rows: {start_pos+batch_size}. Time spent: {time.time() - start_time}')

    processed_dataset = dataset.iloc[:len(vectorized_msg)]
    processed_dataset['CommentMessage'] = vectorized_msg
    save_prepared_dataset(processed_dataset, connection)


if __name__ == "__main__":
    main()
