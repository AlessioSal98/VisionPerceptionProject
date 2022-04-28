import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.models.resnet import BasicBlock
import torchvision.models as models

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class Net(nn.Module):
  def __init__(self,n_classes):
    super().__init__()

    vgg16 = models.vgg16(pretrained=True)
    self.vgg16 = nn.Sequential(vgg16.features,vgg16.avgpool).to(device)

    self.classifier = nn.Sequential(
      nn.Linear(125440, 2048),
      nn.Linear(2048,1024),
      nn.Linear(1024,512),
      #nn.Linear(512,512),
      #nn.Linear(512, 256),
      nn.Linear(512, 256),
      nn.Linear(256, n_classes),
    )
    
    for param in self.vgg16.parameters():
      param.requires_grad =False
  
    for param in self.classifier.parameters():
      param.requires_grad =True
    
  def forward(self, x):
    x = self.feature_extraction(x)
    x = torch.flatten(x,1)
    x = self.classifier(x)
    return x

  def feature_extraction(self,x):
    #Reshape of the video tensors in such a way that they can be passed to the feature extractor, then the tensors are reshaped to their original shape
    frames = []
    for i in range(x.shape[1]):
      f = []
      for j in range(x.shape[0]):
        f.append(x[j][i])
      f = torch.stack(f,0)
      frames.append(f)
    frames = torch.stack(frames,0)

    outputs = []
    for sequence in frames:
      outputs.append(self.vgg16(sequence))
    outputs = torch.stack(outputs,0)
    
    frames = []
    for i in range(outputs.shape[1]):
      f = []
      for j in range(outputs.shape[0]):
        f.append(outputs[j][i])
      f = torch.stack(f,0)
      frames.append(f)
    frames = torch.stack(frames,0)
    return frames