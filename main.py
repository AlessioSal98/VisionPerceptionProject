import trainer
from tqdm import tqdm
#import network
import networkRNN
import dataset_handler
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import os
from sklearn import preprocessing
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from sklearn.metrics import accuracy_score
import argparse
from datetime import datetime
from matplotlib import pyplot as plt
import glob


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

def split(X,y,percentage,seed):
  if(X.shape[0]==y.shape[0]):
    perm = np.random.RandomState(seed=seed).permutation(len(X))
    X=X[perm]
    y=y[perm]
    first_half = int(X.shape[0]*(1-percentage))

    X_1 = X[0:first_half]
    y_1 = y[0:first_half]
    X_2 = X[first_half:X.shape[0]]
    y_2 = y[first_half:y.shape[0]]
  return X_1, y_1 , X_2, y_2

def save(model,args,class_list,history,training_time):
  folder_name = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
  os.mkdir('Saves/'+folder_name)
  save_location=('Saves/'+folder_name+'/')
  torch.save(model, save_location+'Model.pth')
  n_classes = class_list.shape[0]
  classes = ''
  for c in class_list:
    classes = classes+c+'; '

  with open(save_location+'Training_parameters.txt', 'w') as f:
      f.write(('Number of classes: {n_classes}\n'+
                #'Frame merge type: {merge_type}\n'+
                'Number of epochs: {epochs}\n'+
                'Batch Size: {batch_size}\n'+
                'Criterion: CrossEntropy Loss\n'+
                'Optimizer: {optimizer}\n'+
                'Learning Rate: {learning_rate}\n'+
                'Validation Size: {validation_size}\n'+
                'Classes: {classes}\n'+
                'Noise: {noise}\n'+
                'Train Accuracy: {train_accuracy}\n'+
                'Train Loss: {train_loss}\n'+
                'Validation Accuracy: {val_accuracy}\n'+
                'Validation Loss: {val_loss}\n'+
                'Training Time: {training_time}\n\n'+
                'MODEL ARCHITECTURE:\n {model}').format(n_classes=n_classes, 
                                                      #merge_type=args.merge_type, 
                                                      epochs=args.epochs, 
                                                      batch_size = args.batch_size,optimizer = args.optimizer,
                                                      learning_rate = args.learning_rate, validation_size = args.validation_size, 
                                                      classes=classes,noise=args.noise,train_accuracy = round(history['train_accuracy'][-1],3),
                                                      train_loss = round(history['train_loss'][-1],3), val_accuracy = round(history['val_accuracy'][-1],3),
                                                      val_loss = round(history['val_loss'][-1],3),training_time=training_time,model=model))
  plt.plot(history['train_accuracy'],'-o')
  plt.plot(history['val_accuracy'],'-o')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.legend(['Train','Validation'])
  plt.title('Train vs Validation Accuracy')
  plt.savefig(save_location+'accuracy_history.png')
  plt.show()
  plt.plot(history['train_loss'],'-o')
  plt.plot(history['val_loss'],'-o')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend(['Train','Validation'])
  plt.title('Train vs Validation Loss')
  plt.savefig(save_location+'loss_history.png')
  plt.show()

def train(args):
  #n_classes = len(args.classes.split('-'))
  epochs = args.epochs
  batch_size = args.batch_size
  optimizer = args.optimizer
  learning_rate= args.learning_rate
  validation_size = args.validation_size
  #merge_type = args.merge_type
  data_folder_name = args.data_folder_name
  noise = args.noise
  
  X = []
  y = []
  for filename in os.listdir(data_folder_name+'/'):
    if filename.endswith("pt"): 
      X.append(filename)
      y.append(filename.split('_')[1])
  X = np.array(X)
  y = np.array(y)

  classes = np.unique(y)
  n_classes = classes.shape[0]

  
  le = preprocessing.LabelEncoder()
  y = le.fit_transform(y)
  y = torch.from_numpy(y)
  y = F.one_hot(y)
  
  y = y.cpu().detach().numpy()
  X_train, y_train, X_val, y_val = split(X,y,0.2,args.split_seed)

  

  #Model definition
  #model = network.Net(n_classes,merge_type).to(device)

  
  n_frames = torch.load(data_folder_name+'/'+X_train[0]).shape[0]

  #START
  try:
    model = torch.load('Checkpoint/Model.pth')
  except:
    model = networkRNN.netRNN(n_frames=n_frames,n_classes=n_classes)
  #END
  
  
  
  start_time = datetime.now().replace(microsecond=0)
  
  print('#########################################################')
  print('TRAINING INFO')
  print('Classes: ' ,classes)
  print('Number of classes: ',n_classes)
  print('Number of frames: ',n_frames)
  print('Training videos: ',X_train.shape[0])
  #print(y_train.shape)
  print('Validation videos: ', X_val.shape[0])
  #print(y_val.shape)
  print('Start time: ', start_time)
  print('#########################################################')

  history,training_time = trainer.train(model, X_train, y_train , X_val, y_val,
                                            data_folder_name,
                                            n_classes, optimizer, epochs=epochs , 
                                            batch_size=batch_size, validation_size=validation_size,
                                            learning_rate=learning_rate,noise=noise)
  print(training_time)
  save(model,args,classes,history,training_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Parameters')
    #parser.add_argument('--model_folder', dest='model_folder', type=str, help='Indicate the name of the folder that contains the model')
    #parser.add_argument('--num_classes', dest='n_classes', type=int, help='How many classes use during training')
    parser.add_argument('--epochs', dest='epochs', type=int, help='For how many epochs train')
    parser.add_argument('--batch_size', dest='batch_size', type=int, help='How many sample consider for each batch')
    parser.add_argument('--optimizer', dest='optimizer', type=str, help='[SGD] [Adam]')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, help='Learning rate')
    parser.add_argument('--validation_size', dest='validation_size', type=float, help='Validation size')
    #parser.add_argument('--merge_type', dest='merge_type', type=str, help='[early][late] defines when the frames of the video will be merged')
    #parser.add_argument('--classes', dest='classes', type=str, help='Class List written in the form: class1-class2-class3...')
    parser.add_argument('--data_folder_name', dest='data_folder_name', type=str, help='The name of the folder wich contains the train data')
    parser.add_argument('--noise', dest='noise', type=float, help='Define the value of the salt and pepper noise')
    parser.add_argument('--restart_from_checkpoint', dest='restart_from_checkpoint', type=str, help='[True/False] Allow the training to be restarted from the checkpoint')
    parser.add_argument('--split_seed', dest='split_seed', type=int, help='Integer number that identifies the split seed')
    args = parser.parse_args()
    if(args.restart_from_checkpoint == 'False'):
      files = glob.glob('Checkpoint/*')
      for f in files:
         os.remove(f)

    train(args)



