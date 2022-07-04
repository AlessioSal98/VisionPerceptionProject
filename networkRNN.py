import torch
from torch import nn
import torchvision.models as models

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class netRNN(nn.Module):
  def __init__(self,n_frames,n_classes):
        super(netRNN, self).__init__()

        #Definition of the feature extractor (Resnet18)
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(self.feature_extractor.conv1,self.feature_extractor.bn1,
                                               self.feature_extractor.relu,self.feature_extractor.maxpool,
                                               self.feature_extractor.layer1,self.feature_extractor.layer2,
                                               self.feature_extractor.layer3,self.feature_extractor.layer4,
                                               self.feature_extractor.avgpool).to(device)
        #Set of the trainable parameters of the feature extractor to false (it must not be trained)
        for param in self.feature_extractor.parameters():
          param.requires_grad =False

        #Definition of the LSTM module
        self.lstm = nn.LSTM(512, 100).to(device) #512 features per video, 100 output values to be fed into the classifier

        #Definition of the last classifier module
        self.classifier = nn.Sequential(
          nn.Linear(100*n_frames,n_classes),
          #nn.Linear(256,128),
          #nn.Linear(128,128),
          #nn.Linear(128, n_classes), tried different combinations of linear classifier layers but this seems to work better
        ).to(device)
  
  #Forward steps
  def forward(self, i):
    #Feature extraction
    x = i.view(-1, i.shape[2], i.shape[3], i.shape[4])
    x = self.feature_extractor(x)
    x = x.view(i.shape[0], i.shape[1], -1)
    del i
    #LSTM module step
    x, _ = self.lstm(x)
    x = x.view(x.shape[0], -1)
    #Final classification
    x = self.classifier(x)
    return x    