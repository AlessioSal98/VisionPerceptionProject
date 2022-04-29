import utils 
import trainer
from tqdm import tqdm
import network
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

def save(model,args,history):
  folder_name = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
  os.mkdir('Saves/'+folder_name)
  save_location=('Saves/'+folder_name+'/')
  torch.save(model, save_location+'Model.pth')
  with open(save_location+'Training_parameters.txt', 'w') as f:
      f.write(('Number of classes: {n_classes}\n'+
                'Number of epochs: {epochs}\n'+
                'Batch Size: {batch_size}\n'+
                'Criterion: {criterion}\n'+
                'Optimizer: {optimizer}\n'+
                'Learning Rate: {learning_rate}\n'+
                'Validation Size: {validation_size}\n\n'+
                'MODEL ARCHITECTURE:\n {model}').format(n_classes=args.n_classes, epochs=args.epochs, 
                                                      batch_size = args.batch_size, criterion= args.criterion,optimizer = args.optimizer,
                                                      learning_rate = args.learning_rate, validation_size = args.validation_size, model=model))
  
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
  n_classes = args.n_classes
  epochs = args.epochs
  batch_size = args.batch_size
  criterion = args.criterion
  optimizer = args.optimizer
  learning_rate= args.learning_rate
  validation_size = args.validation_size

  
  dataset_folder='Dataset'+str(n_classes)+'Classes/'


  #Video array loading
  X_train = dataset_handler.load(dataset_folder+'X_train.npy')
  X_val = dataset_handler.load(dataset_folder+'X_val.npy')
  y_train = dataset_handler.load(dataset_folder+'y_train.npy')
  y_val = dataset_handler.load(dataset_folder+'y_val.npy')


  #Model definition
  model = network.Net(n_classes).to(device)

  print(X_train.shape)
  print(y_train.shape)
  print(X_val.shape)
  print(y_val.shape)
  history = trainer.train(model, X_train, y_train , X_val, y_val,
                                            n_classes, criterion, optimizer, epochs=epochs , 
                                            batch_size=batch_size, validation_size=validation_size,
                                            learning_rate=learning_rate)
  save(model,args,history)

def test(args):
  model_folder = args.model_folder
  save_path = ('Saves/'+model_folder+'/')
  model = torch.load(save_path+'Model.pth',map_location=torch.device(device))
  model.eval()
  with open(save_path+'Training_parameters.txt') as f:
    content = f.read()
  n_classes = content.split('Number of classes: ')[1].split('\n')[0]
  dataset_folder='Dataset'+str(n_classes)+'Classes/'
  X_test = dataset_handler.load(dataset_folder+'X_test.npy')
  y_test = dataset_handler.load(dataset_folder+'y_test.npy')

  print(X_test.shape)
  print(y_test.shape)
  batch_size = 1
  x_batches = []
  y_batches = []
  for i in range(0, X_test.shape[0], batch_size):
    if(i+batch_size<X_test.shape[0]):
      x_batches.append(X_test[i:i + batch_size])
      y_batches.append(y_test[i:i + batch_size])
  x_batches = np.array(x_batches)
  y_batches = np.array(y_batches)

  accuracies = []
  for i in tqdm(range(x_batches.shape[0])):
    #print('X BATCH:', x_batches[i])
    x = []
    for elem in x_batches[i]:
      x.append(torch.load('Dataset'+str(n_classes)+'Classes/'+elem+'.pt'))
    x = torch.stack(x,0).float().to(device)
    #x = x_batches[i]
    labels = torch.tensor(y_batches[i]).float().to(device)
    outputs = model(x)

    one_hot_encoded_outputs = trainer.one_hot_encode(outputs)

    y_pred = trainer.linearize(one_hot_encoded_outputs)
    y_true = trainer.linearize(labels)

    accuracy = accuracy_score(y_true,y_pred)
    accuracies.append(accuracy)
    del x
    del labels
  accuracies = np.array(accuracies)
  avg_accuracy = np.average(accuracies)
  print('Test accuracy: ',avg_accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Parameters')
    parser.add_argument('--operation', dest='operation', type=str, help='[train][test]')
    parser.add_argument('--model_folder', dest='model_folder', type=str, help='Indicate the name of the folder that contains the model')
    parser.add_argument('--num_classes', dest='n_classes', type=int, help='How many classes use during training')
    parser.add_argument('--epochs', dest='epochs', type=int, help='For how many epochs train')
    parser.add_argument('--batch_size', dest='batch_size', type=int, help='How many sample consider for each batch')
    parser.add_argument('--criterion', dest='criterion', type=str, help='[CrossEntropy] [MSELoss] [BCELoss]')
    parser.add_argument('--optimizer', dest='optimizer', type=str, help='[SGD] [Adam]')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, help='Learning rate')
    parser.add_argument('--validation_size', dest='validation_size', type=float, help='Validation size')
    args = parser.parse_args()
    operation = args.operation
    if(operation == 'train'):
      train(args)
    elif(operation == 'test'):
      test(args)



