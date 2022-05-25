import torch
from torch import nn
import torchvision.models as models

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class netRNN(nn.Module):
  def __init__(self,n_frames,n_classes):
        super(netRNN, self).__init__()

        self.feature_extractor = models.resnet18(pretrained=True)
        #print(self.feature_extractor)
        #self.feature_extractor = nn.Sequential(feature_extractor.features,feature_extractor.avgpool).to(device)
        self.feature_extractor = nn.Sequential(self.feature_extractor.conv1,self.feature_extractor.bn1,
                                               self.feature_extractor.relu,self.feature_extractor.maxpool,
                                               self.feature_extractor.layer1,self.feature_extractor.layer2,
                                               self.feature_extractor.layer3,self.feature_extractor.layer4,
                                               self.feature_extractor.avgpool).to(device)
        for param in self.feature_extractor.parameters():
          param.requires_grad =False

        self.lstm = nn.LSTM(512, 100).to(device) #512 features per video, 100 output values to be fed into the classifier

        self.classifier = nn.Sequential(
          nn.Linear(100*n_frames,n_classes),
          #nn.Linear(256,128),
          #nn.Linear(128,128),
          #nn.Linear(128, n_classes),
        ).to(device)
        #self.fc = nn.Linear(100*n_frames, n_classes).to(device)
        
  def forward(self, i):
    x = i.view(-1, i.shape[2], i.shape[3], i.shape[4])
    x = self.feature_extractor(x)
    x = x.view(i.shape[0], i.shape[1], -1)
    del i
    x, _ = self.lstm(x)
    x = x.view(x.shape[0], -1)
    x = self.classifier(x)
    return x    