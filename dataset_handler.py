'''
This class is used for loading the videos of the dataset and store them as torch tensors, wich will then be used for the network training
'''

import torch
from tqdm import tqdm
from random import sample
import os
import numpy as np
from sklearn import preprocessing
import torch.nn.functional as F
import argparse

#DATASET UTILS IMPORTS
import re
import tempfile
import ssl
import cv2
from urllib import request 

#Utils functions used for downloading the videos
UCF_ROOT = "https://www.crcv.ucf.edu/THUMOS14/UCF101/UCF101/"
_VIDEO_LIST = None
_CACHE_DIR = tempfile.mkdtemp()
unverified_context = ssl._create_unverified_context()

def list_ucf_videos():
  """Lists videos available in UCF101 dataset."""
  global _VIDEO_LIST
  if not _VIDEO_LIST:
    index = request.urlopen(UCF_ROOT, context=unverified_context).read().decode("utf-8")
    videos = re.findall("(v_[\w_]+\.avi)", index)
    _VIDEO_LIST = sorted(set(videos))
  return list(_VIDEO_LIST)

def crop_center_square(frame):
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

def load_video(path, max_frames=0, resize=(224, 224)):
  cap = cv2.VideoCapture(path)
  frames = []
  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      frame = crop_center_square(frame)
      frame = cv2.resize(frame, resize)
      frame = frame[:, :, [2, 1, 0]]
      frames.append(frame)
      
      if len(frames) == max_frames:
        break
  finally:
    cap.release()
  return np.array(frames) / 255.0


def save(array,name):
  with open(name, 'wb') as f:
    np.save(f,array)

def load(name):
  with open(name, 'rb') as f:
    array = np.load(f)
  return array

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset Parameters')
    parser.add_argument('--data_folder_name', dest='data_folder_name', type=str, help='Name of the class that will contain the data')
    parser.add_argument('--classes', dest='classes', type=str, help='Class List written in the form: class1-class2-class3...')
    parser.add_argument('--n_frames', dest='n_frames', type=int, help='How many frames will be taken from each video')
    args = parser.parse_args()

    data_folder_name = args.data_folder_name
    classes = args.classes.split('-')
    n_frames = args.n_frames
    n_classes = len(classes)

    #Creation of the folder
    save_path = (data_folder_name+'/')
    os.mkdir(save_path)

    #Creation of a txt file inside the folder that contains the list of classes
    with open(save_path+'ClassList.txt', 'w') as f:
      f.write(str(classes))
    video_names = list_ucf_videos()

    #Filtering of the videos according to the required classes
    filtered_videos = []
    for video in video_names:
      if video.split('_')[1] in classes:
        filtered_videos.append(video)
    
    video_names = filtered_videos
    print(video_names)
    
    #Generation of the tensors
    for i in tqdm(range(len(video_names))):
      name = video_names[i].split('.avi')[0]
      tensor = load_video(UCF_ROOT+name,0)
      tensor = torch.from_numpy(tensor)
      tensor = torch.swapaxes(tensor, 1 ,3)

      if(n_frames<tensor.shape[0]):
        frames = []
        for j in range (n_frames):
          frames.append(tensor[int((j/n_frames)*tensor.shape[0])])
        tensor = torch.stack(frames,0)

      torch.save(tensor, save_path+name+'.pt')
    