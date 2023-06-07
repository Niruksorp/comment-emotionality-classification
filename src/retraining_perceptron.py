import sys

import torch

import util.model_helper as mh
import util.database as db
from model import logistic_regression
import train
import torch.nn.functional as F
from torch import nn


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


def main(epochs, model_name):
    model = logistic_regression.create_model()
    model = mh.unpack_model(model, db.load_latest_model_by_name(model_name))
    dataset = train.get_and_split_ds()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train.train_model(epochs, model, optimizer, criterion, dataset)
    score = train.evaluate_model(model, dataset)
    train.save_model(model, model_name, score)


if __name__ == "__main__":
    epochs_num = 5
    model_name = 'perceptron'
    if len(sys.argv) > 1:
        epochs_num = int(sys.argv[1])
    if len(sys.argv) > 2:
        model_name = str(sys.argv[2])
    main(epochs_num, model_name)
