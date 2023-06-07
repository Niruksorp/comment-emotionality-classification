import io
import torch


def prepare_model_to_save(model, name, score):
    # Save
    print(f'Start saving model with name: {name} and score: {score}')
    model_out = io.BytesIO()
    torch.save(model.state_dict(), model_out)
    model_out.seek(0)
    byte_array = model_out.read()
    print('Model saved')
    return byte_array, name, score


def unpack_model(model, weights):
    buffer = io.BytesIO(weights)
    model.load_state_dict(torch.load(buffer))
    return model
