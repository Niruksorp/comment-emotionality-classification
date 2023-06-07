import src.util.database as db
import logging
import os
import torch
import time
import numpy as np
import sys

from transformers import AutoTokenizer, AutoModel

DATASET_TABLE_NAME = 'dataset'
PREPROCESSED_DATASET = 'preprocessed_dataset'

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
MAX_SEQUENCE_LENGTH = 2048
model.max_seq_length = MAX_SEQUENCE_LENGTH

batch_size = 16


def get_embs(batch):
    tokens = tokenizer(batch, max_length=MAX_SEQUENCE_LENGTH, padding="max_length", truncation=True)["input_ids"]
    tokens = torch.from_numpy(np.array(tokens))
    with torch.no_grad():
        embeddings = model(tokens)
        embeddings = embeddings.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings


def main(timeout_min):
    # db.clear_preprocessed_data()
    start_ind = db.get_start_ind_for_prepared_ds() + 1
    dataset = db.download_dataset(DATASET_TABLE_NAME)
    start_time = time.time()
    for start_pos in range(0, len(dataset), batch_size):
        if start_pos + batch_size >= len(dataset):
            break
        batch = dataset.iloc[start_pos:start_pos + batch_size]
        text_data = batch['CommentMessage'].tolist()
        batch['CommentMessage'] = get_embs(text_data).numpy().tolist()
        db.save_prepared_dataset(batch, start_ind)
        start_ind += batch_size
        logging.warning(f'Processed rows: {start_pos + batch_size}. Time spent: {time.time() - start_time}')
        if time.time() - start_time > int(timeout_min)*60:
            logging.warning('Finish by timeout')
            break


if __name__ == "__main__":
    timeout = 1
    if len(sys.argv) > 1:
        timeout = int(sys.argv[1])
    main(timeout)
