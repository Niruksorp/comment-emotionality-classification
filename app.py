from curses import flash
from distutils.log import debug

import torch
from flask import Flask, render_template

from src.model.logistic_regression import create_model
from src.util.database import best_model
from src.util.model_helper import unpack_model

app = Flask(__name__)

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

@app.route('/')
def home():
    model = create_model()
    res = best_model()
    model = unpack_model(model, res)
    return render_template('index.html')


if __name__  == '__main__':
    app.run(debug=True)
