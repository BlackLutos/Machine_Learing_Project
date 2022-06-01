#!/usr/bin/env python
# coding: utf-8

# # Lab#3 From Training to Deployment
# 
# 
# ---
# 
# 

# # Copy the dataset from Gdrive to Colab

# In[ ]:


# get_ipython().system('cp -rf drive/MyDrive/MediaTek_IEE5725_Machine_Learning_Lab3/ .')


# # In[ ]:


# from google.colab import drive
# drive.mount('/content/drive')


# # In[ ]:


# get_ipython().system('pip install tensorflow-gpu==1.15')


# # # Unzip your dataset
# # 
# # 

# # In[ ]:


# get_ipython().system('zip -s- /content/MediaTek_IEE5725_Machine_Learning_Lab3/ICME2022_Training_Dataset.zip -O /content/MediaTek_IEE5725_Machine_Learning_Lab3/COMBINED_FILE.zip')
# get_ipython().system('unzip /content/MediaTek_IEE5725_Machine_Learning_Lab3/COMBINED_FILE.zip -d /content/MediaTek_IEE5725_Machine_Learning_Lab3/')


# In[ ]:


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


# In[ ]:


print("Tensorflow Version is %s" % tf.__version__)


# # Data Process

# In[ ]:


class DataLoaderSegmentation(data.Dataset):
    def __init__(self, input_path,label_path,label_name='',transform=None):
        super(DataLoaderSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(input_path,'*.jpg'))
        self.mask_files = []
        self.transforms = transform
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(label_path,os.path.basename(img_path).split('.')[0]+label_name+'.png'))

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            data = cv2.imread(img_path)
            label = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
            label = F.one_hot(torch.from_numpy(label).to(torch.int64),6)
            datalabel = np.concatenate((data,label),axis=2)
            datalabel = np.transpose(datalabel,[2,0,1])
            if self.transforms!=None:
              datalabel = self.transforms(torch.from_numpy(datalabel).float())
            datalabel = np.transpose(datalabel,[1,2,0])
            data = datalabel[:,:,0:3]
            label = datalabel[:,:,3:9]
            return data,label

    def __len__(self):
        return len(self.img_files)


# # Prepare Training dataset

# In[ ]:

print(123)
input_path='ICME2022_Training_Dataset/images'#720/1280
label_path='ICME2022_Training_Dataset/labels/class_labels'
dataset = DataLoaderSegmentation(input_path,label_path,'_lane_line_label_id',transforms.Resize(size=(720,1280)))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=25, shuffle=True)


# In[ ]:


input_path='ICME2022_Training_Dataset/images_real_world'#1080/1920
label_path='ICME2022_Training_Dataset/labels_real_world'
dataset_real = DataLoaderSegmentation(input_path,label_path,'',transforms.Resize(size=(1080,1920)))
dataloader_real = torch.utils.data.DataLoader(dataset_real, batch_size=25, shuffle=True)


# # FIXME#1 Design the Network
# *   input size = 256*256
# *   Channel = 15 and Depth = 5
# *   backbone: conv2d+BN+Relu
# *   output size = 1080*1920
# 
# 
# 
# 

# In[ ]:


inputs = tf.placeholder(tf.float32,shape=(None,None, None, 3))
y_ = tf.placeholder(tf.float32, [None,None, None,6])
x = tf.image.resize_images(inputs, (256, 256))
x = x/255.0
y = tf.image.resize_images(y_, (256, 256))
ch=13
depth=5
xn = []
b=tf.Variable(0.0)
x=tf.layers.conv2d(x,ch,3,1,'same')
x=tf.layers.batch_normalization(x)
x = tf.nn.relu(x)
for i in range(depth):
  xn.append(x)
  x = tf.layers.conv2d(x,ch*(2**(i+1)),3,1,'same')
  x = tf.layers.batch_normalization(x,center=False,scale=False)+b
  x = tf.nn.relu(x)
  x = tf.layers.conv2d(x,ch*(2**(i+1)),3,1,'same')
  x = tf.layers.batch_normalization(x,center=False,scale=False)+b
  x = tf.nn.relu(x)
  if i <depth-1:
    x = tf.nn.avg_pool(x,[1,2,2,1],[1,2,2,1],'SAME')
for i in range(depth):
  if i>0:
    x = tf.keras.layers.UpSampling2D((2,2))(x)
  x = tf.layers.conv2d(x,ch*(2**(depth-i-1)),3,1,'same')+xn[-i-1]
  x = tf.layers.batch_normalization(x,center=False,scale=False)+b
  x = tf.nn.relu(x)
out = tf.layers.conv2d(x,6,3,1,'same')
outputs = out
outputs = tf.image.resize_images(outputs, (1080, 1920))
outputs = tf.argmax(outputs,-1)


# # Set up the Hyper parameters

# In[ ]:


loss=tf.nn.softmax_cross_entropy_with_logits_v2(logits=out,labels=y)
loss=tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.00001)
train = optimizer.minimize(loss+0.0005*b)
saver=tf.train.Saver()
init = tf.global_variables_initializer()


# # Sesstion run and restore Checkpoint

# In[ ]:


sess = tf.Session()
sess.run(init)
saver.restore(sess, 'MediaTek_IEE5725_Machine_Learning_Lab3/model/')


# # Model Profiling to get Flops and #Parameters

# In[ ]:


def stats_graph(graph):
  flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
  params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
  print('FLOPs: {}; Trainable params:{}'.format(flops.total_float_ops, params.total_parameters))
stats_graph(tf.get_default_graph())


# # Set training Epochs, Print the training logs and Save your Checkpoint

# In[ ]:


num_epochs = 10
for epoch in range(num_epochs):
  for i, data in enumerate(dataloader, 0):
    input = data[0].numpy()
    label = data[1].numpy()
    sess.run(train,feed_dict={inputs: input, y_: label})
    if i % 10 == 0:
      print("[%d/%d][%s/%d] loss: %.4f b: %.4f "            %(epoch+1, num_epochs, str(i).zfill(4), len(dataloader), sess.run(loss,feed_dict={inputs: input, y_: label}),sess.run(b)) )
    if i%300==0:
      print('checkpoint saved')
      for i, data in enumerate(dataloader_real, 0):
        input = data[0].numpy()
        label = data[1].numpy()
        sess.run(train,feed_dict={inputs: input, y_: label})
      saver.save(sess, 'MediaTek_IEE5725_Machine_Learning_Lab3/model/')
  saver.save(sess, 'MediaTek_IEE5725_Machine_Learning_Lab3/model/')
  print('checkpoint saved')


# # Save your graph into .pb

# In[ ]:


graph_def = convert_variables_to_constants(sess, sess.graph_def, output_node_names = ['ArgMax'])
tf.train.write_graph(graph_def, 'MediaTek_IEE5725_Machine_Learning_Lab3/', 'lab3_model.pb', as_text = False)


# In[ ]:


plt.imshow(data[0][0]/255.0)


# In[ ]:


plt.imshow(np.argmax(data[1][0], 2))


# # Show your results in different Epochs

# In[ ]:


predict=sess.run(outputs,feed_dict={inputs: data[0]})
plt.imshow(predict[0])


# In[ ]:


sess.close()


# # Run your model on real device

def representative_dataset():
  for data in Path('MediaTek_IEE5725_Machine_Learning_Lab3/Testing_Data_for_Qualification').glob('*.jpg'):
    img = cv2.imread(str(data))
    img = np.expand_dims(img,0)
    img = img.astype(np.float32)
    yield [img]

converter = tf.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file = 'lab3_model.pb',
    input_arrays = ['Placeholder'],
    input_shapes = {'Placeholder':[1, 1080, 1920,3]},
    output_arrays = ['ArgMax'],
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
tflite_model = converter.convert()
open('/content/drive/MyDrive/Colab Notebooks/lab3/lab3_model.tflite', 'wb').write(tflite_model)
