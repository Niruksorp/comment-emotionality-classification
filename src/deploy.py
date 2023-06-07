import torch
import train
from src.model.logistic_regression import create_model
from src.util.model_helper import unpack_model
from util.database import best_model


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


def process():
    model = create_model()
    res = best_model()
    model = unpack_model(model, res)
    model.eval()



if __name__ == "__main__":
    process()
