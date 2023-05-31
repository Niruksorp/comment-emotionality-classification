import os

from torch import softmax

import util.database as db
import util.model_helper as mh
import torch
from tqdm import tqdm
from statistics import mean
import torch.nn.functional as F
import numpy as np
from torchmetrics import Accuracy
# import xgboost as xgb
import pickle
import time
from sklearn.model_selection import train_test_split


df = db.download_dataset('preprocessed_dataset')

train, test = train_test_split(df, test_size=0.2)


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


model_log_reg = LogisticRegression(312, 3)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model_log_reg.parameters(), lr=0.001)
model_log_reg = model_log_reg

Iterations = []
iter = 0
epochs = 10

X_test = torch.tensor(np.array([x[1:-1].split(',') for x in test['CommentMessage'].values]).astype(np.float32))
y_test = torch.tensor(np.array(test['Sentiment'].values.tolist()), dtype=torch.long)
X_train = torch.tensor(np.array([x[1:-1].split(',') for x in train['CommentMessage'].values]).astype(np.float32))
y_train = torch.tensor(np.array(train['Sentiment'].values.tolist()), dtype=torch.long)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
losses = []
losses_test = []
batch_size = 32
acc_test = []
for epoch in tqdm(range(100)):
    cur_loss_idx = len(losses)
    cur_loss_text_idx = len(losses_test)
    permutation = torch.randperm(X_train.size()[0])
    for i in range(0, X_train.size()[0], batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]

        outputs = model_log_reg.forward(batch_x)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        with torch.no_grad():
            output = model_log_reg(X_test)
            loss = criterion(output, y_test)
            losses_test.append(loss.item())
    print("=========================")
    print(f"train loss = {mean(losses[cur_loss_idx:])}")
    print(f"test loss = {mean(losses_test[cur_loss_text_idx:])}")
model_log_reg.eval()

with torch.no_grad():
    res = model_log_reg(X_test)

acc = Accuracy(task="multiclass", num_classes=3)
res_acc = acc(res, y_test).item()
print(f'Accuracy: {res_acc}')

mh.save_model(model_log_reg, 'log_reg', res_acc)

