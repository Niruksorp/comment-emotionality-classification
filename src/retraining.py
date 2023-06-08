import sys

import torch

import util.model_helper as mh
import util.database as db
from model import logistic_regression
import train


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


def main(epochs, model_name):
    model = logistic_regression.create_model()
    model = mh.unpack_model(model, db.load_latest_model_by_name(model_name))
    dataset = train.get_and_split_ds()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train.train_model(epochs, model, optimizer, criterion, dataset)
    score, time = train.evaluate_model(model, dataset)
    train.save_model(model, model_name, score, time)


if __name__ == "__main__":
    epochs_num = 5
    model_name = 'log_reg'
    if len(sys.argv) > 1:
        epochs_num = int(sys.argv[1])
    if len(sys.argv) > 2:
        model_name = str(sys.argv[2])
    main(epochs_num, model_name)
