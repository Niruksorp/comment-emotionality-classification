import torch
import train
from model import perceptron
import sys


def main(epochs):
    model_perceptron = perceptron.create_model()
    dataset = train.get_and_split_ds()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_perceptron.parameters(), lr=0.001)
    train.train_model(epochs, model_perceptron, optimizer, criterion, dataset)
    score, time = train.evaluate_model(model_perceptron, dataset)
    train.save_model(model_perceptron, 'perceptron', score, time)


if __name__ == "__main__":
    epochs_num = 10
    if len(sys.argv) > 1:
        epochs_num = int(sys.argv[1])
    main(epochs_num)
