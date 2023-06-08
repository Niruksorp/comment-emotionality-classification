import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from data_preprocessing import get_embs
from src import train
from util.database import load_latest_model_by_name
from util.model_helper import unpack_model


class Perceptron(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.quant = torch.ao.quantization.QuantStub()
        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        # x = [batch size, height, width]

        batch_size = x.shape[0]

        x = x.view(batch_size, -1)

        # x = [batch size, height * width]
        x = self.quant(x)
        h_1 = self.relu(self.input_fc(x))

        # h_1 = [batch size, 250]

        h_2 = self.relu2(self.hidden_fc(h_1))

        # h_2 = [batch size, 100]

        y_pred = self.relu3(self.output_fc(h_2))

        y_pred = self.dequant(y_pred)
        # y_pred = [batch size, output dim]
        return F.softmax(y_pred)

    def get_modules_to_fuse(self):
        return [['input_fc', 'relu'], ['hidden_fc', 'relu2'], ['output_fc', 'relu3']]


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.linear(x)
        outputs = self.relu(x)
        outputs = self.dequant(outputs)
        outputs = F.softmax(outputs)
        return outputs

    def get_modules_to_fuse(self):
        return [['linear', 'relu']]


def get_emotional_grade(class_number):
    if class_number == 0:
        return 'negative'
    if class_number == 1:
        return 'neutral'
    if class_number == 2:
        return 'positive'


def create_model_log_reg():
    return LogisticRegression(312, 3)


def create_model_perceptron():
    return Perceptron(312, 3)


def get_model_by_name(name):
    if name == 'perceptron':
        return create_model_perceptron()
    elif name == 'log_reg':
        return create_model_log_reg()
    return None


def eval(model, text):
    model.eval()
    emb = get_embs([text])
    with torch.no_grad():
        res = model(emb)
    return get_emotional_grade(np.argmax(res.numpy()))


def quantize_model(model, dataset):
    model.eval()
    model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    model_fp32_fused = torch.ao.quantization.fuse_modules(model, model.get_modules_to_fuse())
    model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)
    X_test, y_test, _, _ = dataset
    input_fp32 = X_test
    model_fp32_prepared(input_fp32)
    model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
    return model_int8


def main():
    name = 'log_reg'
    res = load_latest_model_by_name(name)
    model = get_model_by_name(name)
    model = unpack_model(model, res)
    dataset = train.get_and_split_ds()
    model_int8 = quantize_model(model, dataset)
    acc, time = train.evaluate_model(model_int8, dataset)
    train.save_model(model_int8, name + '_q', acc, time)


if __name__ == '__main__':
    main()
