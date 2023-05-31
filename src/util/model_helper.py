import torch
import src.util.database as db

PATH = "../entire_model.pt"


def save_model(model, name, score):
    # Save
    print(f'Start saving model with name: {name} and score: {score}')
    torch.save(model, PATH)
    byte_array = b''
    with open(PATH, "rb") as f:
        while byte := f.read(1):
            byte_array += byte
    db.save_model(byte_array, name, score)
    print('Model saved')
