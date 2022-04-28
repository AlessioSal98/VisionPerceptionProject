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


UCF_ROOT = "https://www.crcv.ucf.edu/THUMOS14/UCF101/UCF101/"


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

def train(model , X_train, y_train , X_val, y_val,n_classes, criterion ,optimizer ,epochs ,batch_size,validation_size, learning_rate):

  #Feature Extractor Definition
  vgg16 = models.vgg16(pretrained=True)
  for param in vgg16.parameters():
    param.requires_grad =False
  vgg16 = nn.Sequential(vgg16.features,vgg16.avgpool).to(device)
  #print(vgg16)  

  #Training parameters definition
  criterion = get_loss(criterion)
  optimizer = get_optimizer(model,optimizer,learning_rate)
  epochs = epochs
  batch_size= batch_size

  train_accuracy_history = []
  val_accuracy_history = []
  train_loss_history = []
  val_loss_history = []
  for epoch in range(epochs):
    #Train and Validation batches creation
    #Train Batches
    x_batches = []
    y_batches = []
    for i in range(0, X_train.shape[0], batch_size):
      if(i+batch_size<X_train.shape[0]):
        x_batches.append(X_train[i:i + batch_size])
        y_batches.append(y_train[i:i + batch_size])
    x_batches_train = np.array(x_batches)
    y_batches_train = np.array(y_batches)

    #Validation Batches
    x_batches = []
    y_batches = []
    for i in range(0, X_val.shape[0], batch_size):
      if(i+batch_size<X_val.shape[0]):
        x_batches.append(X_val[i:i + batch_size])
        y_batches.append(y_val[i:i + batch_size])
    x_batches_val = np.array(x_batches)
    y_batches_val = np.array(y_batches)

    #Execution of the train and validation steps, with relative collection of the losses
    train_loss,train_accuracy = step(x_batches_train,y_batches_train,n_classes,vgg16,optimizer,criterion,model,True)
    val_loss,val_accuracy = step(x_batches_val,y_batches_val,n_classes,vgg16,optimizer,criterion,model,False)

    train_accuracy_history.append(train_accuracy)
    val_accuracy_history.append(val_accuracy)
    train_loss_history.append(train_loss.cpu().detach().numpy())
    val_loss_history.append(val_loss.cpu().detach().numpy())
    
    print('Epoch {} of {}, Train Loss: {:.3f}, Validation Loss: {:.3f}, Train Accuracy: {:.5f}, Validation Accuracy: {:.5f}'.format(epoch+1, epochs, train_loss, val_loss, train_accuracy, val_accuracy))
  history = {
    'train_accuracy': train_accuracy_history,
    'val_accuracy': val_accuracy_history,
    'train_loss': train_loss_history,
    'val_loss': val_loss_history
  }


  print('Finished Training')
  return history

def step(x_batches,y_batches,n_classes,feature_extractor,optimizer,criterion,model,isTrain):
  #Loss initialization
  total_loss = 0
  #For each batch in the set of batches
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
      x.append(torch.load('Dataset'+str(n_classes)+'Classes/'+elem+'.pt'))
    x = torch.stack(x,0).float().to(device)

    
    #x = x_batches[i]
    labels = torch.tensor(y_batches[i]).float().to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(x)
    loss = criterion(outputs, labels)

    if(isTrain==False):
      loss.backward()
      optimizer.step()

    total_loss = total_loss+loss

    one_hot_encoded_outputs = one_hot_encode(outputs)

    y_pred = linearize(one_hot_encoded_outputs)
    y_true = linearize(labels)

    accuracy = accuracy_score(y_true,y_pred)
    accuracies.append(accuracy)

    del x
    del labels
  accuracies = np.array(accuracies)
  avg_accuracy = np.average(accuracies)

  return total_loss,avg_accuracy

def test(model,X_test,y_test,n_classes):
  model.eval()
  x = []
  for elem in X_test:
    x.append(torch.load('Dataset'+str(n_classes)+'Classes/'+elem+'.pt'))
  x = torch.stack(x,0).float().to(device)
  y_test = torch.tensor(y_test).float().to(device)

  outputs = model(x)
  one_hot_encoded_outputs = one_hot_encode(outputs)

  y_pred = linearize(one_hot_encoded_outputs)
  y_true = linearize(y_test)

  print('Predictions: ',y_pred)
  print('True: ', y_true)
  accuracy = accuracy_score(y_true,y_pred)
  return accuracy

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

def get_loss(criterion):
  if criterion=='MSELoss':
    return nn.MSELoss()
  elif criterion == 'BCELoss':
    return nn.BCELoss()
  elif criterion == 'CrossEntropy':
    return nn.CrossEntropyLoss()
  elif criterion == 'KLD':
    return nn.KLDivLoss()
  else:
    return nn.CrossEntropyLoss
