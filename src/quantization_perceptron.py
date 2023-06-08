import quantization as qu
import torch
from torch import nn
import torch.nn.functional as F
from model import perceptron
from data_preprocessing import get_embs
from model.logistic_regression import create_model
from model import perceptron
import train
from util.database import load_latest_model_by_name
from util.model_helper import unpack_model
from model import logistic_regression
import numpy as np

def main():
    name = 'perceptron'
    res = load_latest_model_by_name(name)
    model = qu.get_model_by_name(name)
    model = unpack_model(model, res)
    dataset = train.get_and_split_ds()
    model_int8 = qu.quantize_model(model, dataset)
    acc, time = train.evaluate_model(model_int8, dataset)
    train.save_model(model_int8, name+'_q', acc, time)


if __name__ == '__main__':
    main()
