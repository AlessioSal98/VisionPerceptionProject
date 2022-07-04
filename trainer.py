import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import torchvision.models as models
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from skimage.util import random_noise
import os
from datetime import datetime,timedelta

UCF_ROOT = "https://www.crcv.ucf.edu/THUMOS14/UCF101/UCF101/"


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model , X_train, y_train , X_val, y_val,data_folder_name,n_classes ,optimizer ,epochs ,batch_size,validation_size, learning_rate,noise):

  #Training parameters definition
  criterion = nn.CrossEntropyLoss()
  optimizer = get_optimizer(model,optimizer,learning_rate)
  epochs = epochs
  batch_size= batch_size


  train_accuracy_history = []
  val_accuracy_history = []
  train_loss_history = []
  val_loss_history = []

  

  #Loading of the checkpoints (or start from 0 in case of no checkpoints available)
  try:
    with open('Checkpoint/currentEpoch.txt', 'r') as f:
      for line in f:
        start_epoch = (int(line.strip()))
  except:
    start_epoch=0
  print('Starting train from epoch:',start_epoch)

  try:
    with open('Checkpoint/currentTime.txt', 'r') as f:
      for line in f:
        t = datetime.strptime(line.strip(),"%H:%M:%S")
        total_time = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
  except:
    total_time = datetime.now().replace(microsecond=0)-datetime.now().replace(microsecond=0)
  
  #Load of accuracies/losses according to the checkpoint
  train_accuracy_history = load_history('train_accuracy_history')
  val_accuracy_history = load_history('val_accuracy_history')
  train_loss_history = load_history('train_loss_history')
  val_loss_history = load_history('val_loss_history')

  #Training start
  for epoch in range(start_epoch,epochs):

    start_time = datetime.now().replace(microsecond=0)
    #Train and Validation batches creation
    #Train Batches
    x_batches = []
    y_batches = []

    perm = np.random.permutation(len(X_train))
    X_train=X_train[perm]
    y_train=y_train[perm]
    for i in range(0, X_train.shape[0], batch_size):
      if(i+batch_size<X_train.shape[0]):
        x_batches.append(X_train[i:i + batch_size])
        y_batches.append(y_train[i:i + batch_size])
    x_batches_train = np.array(x_batches)
    y_batches_train = np.array(y_batches)

    #Validation Batches
    x_batches = []
    y_batches = []

    perm = np.random.permutation(len(X_val))
    X_val=X_val[perm]
    y_val=y_val[perm]
    for i in range(0, X_val.shape[0], batch_size):
      if(i+batch_size<X_val.shape[0]):
        x_batches.append(X_val[i:i + batch_size])
        y_batches.append(y_val[i:i + batch_size])
    x_batches_val = np.array(x_batches)
    y_batches_val = np.array(y_batches)

    #Execution of the train and validation steps, with the relative collection of the losses
    train_loss,train_accuracy = step(x_batches_train,y_batches_train,data_folder_name,n_classes,optimizer,criterion,model,noise,True)
    val_loss,val_accuracy = step(x_batches_val,y_batches_val,data_folder_name,n_classes,optimizer,criterion,model,noise,False)


    train_accuracy_history.append(train_accuracy)
    val_accuracy_history.append(val_accuracy)
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    
    #Computation of the time needed to complete the epoch and summing with the current total time
    end_time = datetime.now().replace(microsecond=0)
    epoch_time = end_time-start_time
    total_time = total_time+epoch_time

    #Update of the checkpoint informations
    with open('Checkpoint/currentEpoch.txt', 'w') as f:
      f.write(str(epoch+1))
    with open('Checkpoint/currentTime.txt', 'w') as f:
      f.write(str(total_time))
    torch.save(model, 'Checkpoint/Model.pth')
    save_history('train_accuracy_history',train_accuracy_history)
    save_history('val_accuracy_history',val_accuracy_history)
    save_history('train_loss_history',train_loss_history)
    save_history('val_loss_history',val_loss_history)

    #Print of the results of the current epoch
    print('Epoch {} of {}, Train Loss: {:.3f}, Validation Loss: {:.3f}, Train Accuracy: {:.5f}, Validation Accuracy: {:.5f}'.format(epoch+1, epochs, train_loss, val_loss, train_accuracy, val_accuracy))
  #Generation of the history of accuracies and losses
  history = {
    'train_accuracy': train_accuracy_history,
    'val_accuracy': val_accuracy_history,
    'train_loss': train_loss_history,
    'val_loss': val_loss_history
  }

  print('Finished Training')
  return history,total_time

#Methods for save and load accuracies and validation histories
def save_history(filename,history):
  with open('Checkpoint/'+filename+'.txt', 'w') as f:
    for i in history:
      f.write(str(i)+'\n')

def load_history(filename):
  history = []
  try:
    with open('Checkpoint/'+filename+'.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
          history.append(float(line))
  except:
    history.append(0)
  return history

#Training/Evaluation step
def step(x_batches,y_batches,data_folder_name,n_classes,optimizer,criterion,model,noise,isTrain):
  #Loss initialization
  total_loss = 0
  #For each batch in the set of batches
  losses = []
  accuracies = []
  correct = 0
  if (isTrain==True):
    model.train()
  else:
    model.eval()

  #Iteration of the batches
  for i in tqdm(range(x_batches.shape[0])):
    x = []
    for elem in x_batches[i]:
      video = torch.load(data_folder_name+'/'+elem)
      x.append(torch.from_numpy(random_noise(video, mode='salt', amount=noise)))
    x = torch.stack(x,0).float().to(device)

    
    labels = torch.tensor(y_batches[i]).float().to(device)

    #Set to zero the parameter gradients
    optimizer.zero_grad()

    #Feeding the model with the input samples
    outputs = model(x)
    #Computation of the loss according to the outputs
    loss = criterion(outputs, labels)

    #Updating of the weights if the network is in training step
    if(isTrain==True):
      loss.backward()
      optimizer.step()

    losses.append(loss.item())

    #Accuracy computation
    one_hot_encoded_outputs = one_hot_encode(outputs)
    y_pred = linearize(one_hot_encoded_outputs)
    y_true = linearize(labels)
    accuracy = accuracy_score(y_true,y_pred)
    accuracies.append(accuracy)

    del x
    del labels
  losses = np.array(losses)
  avg_loss = np.average(losses)
  accuracies = np.array(accuracies)
  avg_accuracy = np.average(accuracies)

  return avg_loss,avg_accuracy

#Method that creates the one hot encode of the outputs of the network
def one_hot_encode(x):
  hot_encoded = []
  for elem in x:
    argmax = int(torch.argmax(elem,dim=0))
    encode = torch.zeros(1,elem.shape[0])
    encode[0][argmax]=1
    hot_encoded.append(encode)
  hot_encoded = torch.stack(hot_encoded,0).to(device)
  return hot_encoded

def linearize(x):
  res = []
  for elem in x:
    res.append(elem.cpu().argmax())
  res = np.array(res)
  return res

#Method used for the optimized definition
def get_optimizer(model,optimizer,learning_rate):
  if optimizer == 'Adam':
    return optim.Adam(model.parameters(), lr=learning_rate)
  else:
    return optim.SGD(model.parameters(), lr = learning_rate)
