'''
This class is used for loading the videos of the dataset and store them as torch tensors, wich will then be used for the network training
'''

import torch
import utils
from tqdm import tqdm
from random import sample
import os
import numpy as np
from sklearn import preprocessing
import torch.nn.functional as F

UCF_ROOT = "https://www.crcv.ucf.edu/THUMOS14/UCF101/UCF101/"

def create_tensors(save_path,n_classes,n_frames=5):
  os.mkdir(save_path)
  video_names = utils.list_ucf_videos()
  #video_names = sample(utils.list_ucf_videos(),dataset_size)
  labels = []
  for name in video_names:
    labels.append(name.split('_')[1])
  classes = list(set(labels))[:n_classes]
  print(classes)

  filtered_videos = []
  for video in video_names:
    if video.split('_')[1] in classes:
    #if 'Archery' in video or 'Biking' in video or 'Drumming' in video or 'HorseRace' in video or 'Punch' in video:
      filtered_videos.append(video)
  
  #print(filtered_videos)
  video_names = filtered_videos
  print(video_names)
  
  #video_names = sample(video_names,dataset_size)

  for i in tqdm(range(len(video_names))):
    name = video_names[i].split('.avi')[0]
    tensor = utils.load_video(UCF_ROOT+name,n_frames)
    tensor = torch.from_numpy(tensor)
    tensor = torch.swapaxes(tensor, 1 ,3)
    torch.save(tensor, save_path+name+'.pt')

def create_train_test_split(dataset_folder):
  videos = []
  video_names = []
  video_labels = []
  for filename in os.listdir(dataset_folder):
      if filename.endswith("pt"): 
          name = filename.split('.pt')[0]
          video_labels.append(name.split('_')[1])
          video_names.append(name)

  video_names = np.array(video_names)
  #Label 1 hot encoding
  le = preprocessing.LabelEncoder()
  video_labels = le.fit_transform(video_labels)
  video_labels = torch.from_numpy(video_labels)
  video_labels = F.one_hot(video_labels)

  video_labels = video_labels.cpu().detach().numpy()

  unique, counts = np.unique(video_labels, return_counts=True)

  X_train, y_train , X_test, y_test = split(video_names,video_labels,0.2)
  X_train, y_train , X_val, y_val = split(X_train,y_train,0.3)

  save(X_train,dataset_folder+'X_train.npy')
  save(y_train,dataset_folder+'y_train.npy')
  save(X_val,dataset_folder+'X_val.npy')
  save(y_val,dataset_folder+'y_val.npy')
  save(X_test,dataset_folder+'X_test.npy')
  save(y_test,dataset_folder+'y_test.npy')


def save(array,name):
  with open(name, 'wb') as f:
    np.save(f,array)

def load(name):
  with open(name, 'rb') as f:
    array = np.load(f)
  return array


def split(X,y,percentage):
  if(X.shape[0]==y.shape[0]):
    perm = np.random.permutation(len(X))
    X=X[perm]
    y=y[perm]
    first_half = int(X.shape[0]*(1-percentage))

    X_1 = X[0:first_half]
    y_1 = y[0:first_half]
    X_2 = X[first_half:X.shape[0]]
    y_2 = y[first_half:y.shape[0]]
  return X_1, y_1 , X_2, y_2
