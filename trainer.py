import torch
import torch.nn as nn
import torch.optim as optim
import utils
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import torchvision.models as models
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from skimage.util import random_noise


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
  train_accuracy_history.append(0)
  val_accuracy_history.append(0)
  train_loss_history.append(0)
  val_loss_history.append(0)

  for epoch in range(epochs):
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

    #Execution of the train and validation steps, with relative collection of the losses
    train_loss,train_accuracy = step(x_batches_train,y_batches_train,data_folder_name,n_classes,optimizer,criterion,model,noise,True)
    val_loss,val_accuracy = step(x_batches_val,y_batches_val,data_folder_name,n_classes,optimizer,criterion,model,noise,False)

    train_accuracy_history.append(train_accuracy)
    val_accuracy_history.append(val_accuracy)
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    
    print('Epoch {} of {}, Train Loss: {:.3f}, Validation Loss: {:.3f}, Train Accuracy: {:.5f}, Validation Accuracy: {:.5f}'.format(epoch+1, epochs, train_loss, val_loss, train_accuracy, val_accuracy))
  history = {
    'train_accuracy': train_accuracy_history,
    'val_accuracy': val_accuracy_history,
    'train_loss': train_loss_history,
    'val_loss': val_loss_history
  }


  print('Finished Training')
  return history

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
  for i in tqdm(range(x_batches.shape[0])):
    #print('X BATCH:', x_batches[i])
    x = []
    for elem in x_batches[i]:
      video = torch.load(data_folder_name+'/'+elem)
      x.append(torch.from_numpy(random_noise(video, mode='salt', amount=noise)))
      #x.append(torch.load('Dataset/'+elem))
      #x.append(torch.load('Dataset'+str(n_classes)+'Classes/'+elem+'.pt'))
    x = torch.stack(x,0).float().to(device)
    #x = x/255

    
    #x = x_batches[i]
    labels = torch.tensor(y_batches[i]).float().to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(x)
    loss = criterion(outputs, labels)

    if(isTrain==True):
      loss.backward()
      optimizer.step()

    #total_loss = total_loss+loss
    losses.append(loss.item())

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

def get_optimizer(model,optimizer,learning_rate):
  if optimizer == 'Adam':
    return optim.Adam(model.parameters(), lr=learning_rate)
  else:
    return optim.SGD(model.parameters(), lr = learning_rate)
