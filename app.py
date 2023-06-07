import torch
from torch import nn
from flask import Flask, render_template
from src.data_preprocessing import get_embs
from src.model.logistic_regression import create_model
from src.model import perceptron
from src.util.database import best_model
from src.util.model_helper import unpack_model
import torch.nn.functional as F
import numpy as np

app = Flask(__name__)


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


class Perceptron(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)

    def forward(self, x):
        # x = [batch size, height, width]

        batch_size = x.shape[0]

        x = x.view(batch_size, -1)

        # x = [batch size, height * width]

        h_1 = F.relu(self.input_fc(x))

        # h_1 = [batch size, 250]

        h_2 = F.relu(self.hidden_fc(h_1))

        # h_2 = [batch size, 100]

        y_pred = F.softmax(self.output_fc(h_2))

        # y_pred = [batch size, output dim]

        return y_pred


def get_emotional_grade(class_number):
    if class_number == 0:
        return 'negative'
    if class_number == 1:
        return 'neutral'
    if class_number == 2:
        return 'positive'

def eval(model, text):
    model.eval()
    emb = get_embs([text])[0]
    with torch.no_grad():
        res = model(emb)
    return get_emotional_grade(np.argmax(res.numpy()))

def get_model_by_name(name):
    if name == 'perceprton':
        return perceptron.create_model()
    elif name == 'log_reg':
        return create_model()
    return None

@app.route('/')
def home():
    res, name = best_model()
    model = get_model_by_name(name)
    if model is None:
        return 'BRUH'
    model = unpack_model(model, res)
    text = 'Fucking text'
    print(f'This comment is {eval(model, text)}')
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
    # res, name = best_model()
    # model = get_model_by_name(name)
    # model = unpack_model(model, res)
    # text = 'Fucking text'
    # print(f'This comment is {eval(model, text)}')
