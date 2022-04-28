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

def save(model,params,history):
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
                'MODEL ARCHITECTURE:\n {model}').format(n_classes=n_classes, epochs=epochs, 
                                                      batch_size = batch_size, criterion= criterion,optimizer = optimizer,
                                                      learning_rate = learning_rate, validation_size = validation_size, model=model))
  
  plt.plot(history['train_accuracy'],'-o')
  plt.plot(history['val_accuracy'],'-o')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.legend(['Train','Validation'])
  plt.title('Train vs Validation Accuracy')
  plt.savefig(save_location+'accuracy_history.png')
  plt.plot(history['train_loss'],'-o')
  plt.plot(history['val_loss'],'-o')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend(['Train','Validation'])
  plt.title('Train vs Validation Loss')
  plt.savefig(save_location+'loss_history.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Parameters')
    parser.add_argument('--num_classes', dest='n_classes', type=int, help='How many classes use during training')
    parser.add_argument('--epochs', dest='epochs', type=int, help='For how many epochs train')
    parser.add_argument('--batch_size', dest='batch_size', type=int, help='How many sample consider for each batch')
    parser.add_argument('--criterion', dest='criterion', type=str, help='[CrossEntropy] [MSELoss] [BCELoss]')
    parser.add_argument('--optimizer', dest='optimizer', type=str, help='[SGD] [Adam]')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, help='Learning rate')
    parser.add_argument('--validation_size', dest='validation_size', type=float, help='Validation size')
    args = parser.parse_args()
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
    X_test = dataset_handler.load(dataset_folder+'X_test.npy')
    y_train = dataset_handler.load(dataset_folder+'y_train.npy')
    y_val = dataset_handler.load(dataset_folder+'y_val.npy')
    y_test = dataset_handler.load(dataset_folder+'y_test.npy')


    #Model definition
    model = network.Net(n_classes).to(device)

    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    print(X_test.shape)
    print(y_test.shape)
    history = trainer.train(model, X_train, y_train , X_val, y_val,
                                              n_classes, criterion, optimizer, epochs=epochs , 
                                              batch_size=batch_size, validation_size=validation_size,
                                              learning_rate=learning_rate)
    save(model,args,history)
    
    #TEST

    #model = torch.load('Model'+str(n_classes)+'Classes.pth',map_location=torch.device(device))
    #accuracy = trainer2.test(model,X_test,y_test,n_classes)
    #print('TEST ACCURACY: ', accuracy)



