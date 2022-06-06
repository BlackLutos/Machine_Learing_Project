import cv2
import matplotlib.pyplot as plt
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import glob
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import tensorflow as tf

from pathlib import Path
from tensorflow.python.framework.graph_util import convert_variables_to_constants

def representative_dataset():
  for data in Path('MediaTek_IEE5725_Machine_Learning_Lab3/Testing_Data_for_Qualification').glob('*.jpg'):
    img = cv2.imread(str(data))
    img = np.expand_dims(img,0)
    img = img.astype(np.float32)
    yield [img]

converter = tf.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file = 'MediaTek_IEE5725_Machine_Learning_Lab3/lab3_model.pb',
    input_arrays = ['Placeholder'],
    input_shapes = {'Placeholder':[1, 1080, 1920,3]},
    output_arrays = ['ArgMax'],
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
tflite_model = converter.convert()
open('MediaTek_IEE5725_Machine_Learning_Lab3/lab3_model.tflite', 'wb').write(tflite_model)