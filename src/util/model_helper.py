import torch

PATH = "../entire_model.pt"


def prepare_model_to_save(model, name, score):
    # Save
    print(f'Start saving model with name: {name} and score: {score}')
    torch.save(model, PATH)
    byte_array = b''
    with open(PATH, "rb") as f:
        while byte := f.read(1):
            byte_array += byte
    print('Model saved')
    return byte_array, name, score
