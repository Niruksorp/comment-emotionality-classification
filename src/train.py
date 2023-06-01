import sys
from statistics import mean

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy
from tqdm import tqdm

import util.database as db
import util.model_helper as mh
from model import logistic_regression


def get_and_split_ds():
    df = db.download_dataset('preprocessed_dataset')

    train, test = train_test_split(df, test_size=0.2)
    X_test = torch.tensor(np.array([x[1:-1].split(',') for x in test['CommentMessage'].values]).astype(np.float32))
    y_test = torch.tensor(np.array(test['Sentiment'].values.tolist()), dtype=torch.long)
    X_train = torch.tensor(np.array([x[1:-1].split(',') for x in train['CommentMessage'].values]).astype(np.float32))
    y_train = torch.tensor(np.array(train['Sentiment'].values.tolist()), dtype=torch.long)

    return X_test, y_test, X_train, y_train


def train_model(epochs, model, optimizer, criterion, dataset):
    X_test, y_test, X_train, y_train = dataset
    losses = []
    losses_test = []
    batch_size = 32
    for epoch in tqdm(range(epochs)):
        cur_loss_idx = len(losses)
        cur_loss_text_idx = len(losses_test)
        permutation = torch.randperm(X_train.size()[0])
        for i in range(0, X_train.size()[0], batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            outputs = model.forward(batch_x)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            with torch.no_grad():
                output = model(X_test)
                loss = criterion(output, y_test)
                losses_test.append(loss.item())
        print("=========================")
        print(f"Epoch: {epoch}")
        print(f"train loss = {mean(losses[cur_loss_idx:])}")
        print(f"test loss = {mean(losses_test[cur_loss_text_idx:])}")


def evaluate_model(model, dataset):
    X_test, y_test, _, _ = dataset
    model.eval()

    with torch.no_grad():
        res = model(X_test)

    acc = Accuracy(task="multiclass", num_classes=3)
    res_acc = acc(res, y_test).item()
    print(f'Accuracy: {res_acc}')
    return res_acc


def save_model(model, name, score):
    prepared = mh.prepare_model_to_save(model, name, score)
    db.save_model(*prepared)


def main(epochs):
    model_log_reg = logistic_regression.create_model()
    dataset = get_and_split_ds()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_log_reg.parameters(), lr=0.001)
    train_model(epochs, model_log_reg, optimizer, criterion, dataset)
    score = evaluate_model(model_log_reg, dataset)
    save_model(model_log_reg, 'log_reg', score)


if __name__ == "__main__":
    epochs_num = 10
    if len(sys.argv) > 1:
        epochs_num = int(sys.argv[1])
    main(epochs_num)
