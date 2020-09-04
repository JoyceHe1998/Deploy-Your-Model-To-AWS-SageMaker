import json
import logging
import os
import torch
import numpy as np
from torchvision import transforms
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def mse(imageA, imageB):
  return -torch.mean((imageA - imageB)**2) * 10000000000000

def calculate_diagonals(matrix, sequence_length):
  arr = torch.zeros([sequence_length, sequence_length * 2 - 1])
  for i in range(sequence_length):
    arr[i, (sequence_length - 1 - i):(sequence_length * 2 - 1 - i)] = matrix[i, :]
  sum = arr.sum(dim = 0)

  # divide by array: e.g. when sequence_length = 10, divide_by_arr = torch.Tensor([1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1])
  second_half = np.arange(1, sequence_length + 1)[::-1].copy()
  second_half = torch.from_numpy(second_half)
  first_half = torch.arange(1, sequence_length)

  divide_by_arr = torch.cat((first_half, second_half))
  result = sum / divide_by_arr
  return result

class Classification(nn.Module):
  def __init__(self, sequence_length):
    super(Classification, self).__init__()
    self.sequence_length = sequence_length

  def forward(self, clip1, clip2):
    matrix = torch.zeros([self.sequence_length, self.sequence_length])
    xvalues = torch.arange(0, self.sequence_length)
    yvalues = torch.arange(0, self.sequence_length)

    xx, yy = torch.meshgrid(xvalues, yvalues)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)

    for i in range(len(xx)):
        matrix[xx[i]][yy[i]] = mse(clip1[yy[i]], clip2[xx[i]])

    return calculate_diagonals(matrix, self.sequence_length)[int(self.sequence_length / 2 - 1) : int(self.sequence_length * 3 / 2)]

class ToySynNet_Points(nn.Module):
    def __init__(self, clip_length, n_continuous, layer_neurons=[120,84]):
        super().__init__()
        self.clip_length = clip_length
        self.n_continuous = n_continuous
        self.fc1 = nn.Linear(n_continuous, layer_neurons[0])
        self.fc2 = nn.Linear(layer_neurons[0], layer_neurons[1])

        self.classification = Classification(self.clip_length)

    def encode(self, X):
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return X

    def forward(self, clip1, clip2):
        clip1 = self.encode(clip1)
        clip2 = self.encode(clip2)
        clip1 = torch.squeeze(clip1)
        clip2 = torch.squeeze(clip2)
        classification = self.classification(clip1, clip2)
        return classification

def model_fn(model_dir):
    model = ToySynNet_Points(10, 4)
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=torch.device('cpu')))
    return model


def input_fn(request_body, content_type='application/json'):
    logger.info('Deserializing the input data.')
    # if content_type == 'application/json':
    input_data = json.loads(request_body)

    # clip_cam1 and clip_cam2 of type: <class 'list'>
    clip_cam1 = input_data['clip_cam1']
    clip_cam2 = input_data['clip_cam2']

    clip_cam1 = torch.FloatTensor(clip_cam1)
    clip_cam2 = torch.FloatTensor(clip_cam2)

    return (clip_cam1, clip_cam2)


def predict_fn(input_data, model):
    logger.info('Generating prediction based on input parameters.')
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        y_pred = model(input_data[0].to(device), input_data[1].to(device))
        y_pred = torch.unsqueeze(y_pred, 0)
        predicted = torch.max(y_pred.data, 1)[1].item() - 10/2 # 10 is the length of the video clip
        return predicted


def output_fn(prediction_output, accept='application/json'):
    result = {'predicted time offset': prediction_output}

    # if accept == 'application/json':
    return json.dumps(result), accept