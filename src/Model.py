import matplotlib.pyplot as plt
import numpy as np
import torch.nn
from torch import nn
import torch.nn.functional as F

class Sia_net(nn.Module):

    def __init__(self,mnist=True):
        super().__init__()
        self.mnist = mnist

        if self.mnist:
            padding = 1
            kernel = 3
            batch = np.array([4,8,8])
            para = np.array([[1, 4], [4, 8], [8, 8]])
            fc_para = np.array([[6272, 1000],[1000, 100],[100, 5]])
        else:
            padding = 2
            kernel = 5
            batch = np.array([32, 32, 64])
            para = np.array([[3, 32], [32, 32], [32, 64]])
            fc_para = np.array([[65536,1000],[1000,1000],[1000,5]])

        self.cnn = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(para[0][0].item(),para[0][1].item(), kernel),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(batch[0].item()),

            nn.ReflectionPad2d(padding),
            nn.Conv2d(para[1][0].item(),para[1][1].item(), kernel),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(batch[1].item()),

            nn.ReflectionPad2d(padding),
            nn.Conv2d(para[2][0].item(),para[2][1].item(), kernel),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(batch[2].item())
        )

        self.fc = nn.Sequential(
            nn.Linear(fc_para[0][0].item(), fc_para[0][1].item()),
            nn.ReLU(inplace=True),
            nn.Linear(fc_para[1][0].item(), fc_para[1][1].item()),
            nn.ReLU(inplace=True),
            nn.Linear(fc_para[2][0].item(), fc_para[2][1].item()),
        )

    def forward_once(self,x):
        output = self.cnn(x)
        output = output.view(output.size()[0],-1)
        return self.fc(output)

    def forward(self,input1,input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once((input2))
        return output1,output2



class ContrastiveLoss(nn.Module):

    def __init__(self,margin=2.0):
        super(ContrastiveLoss,self).__init__()
        self.margin = margin

    def forward(self,output1,output2,label):
        euclidean_distance = F.pairwise_distance(output1,output2,keepdim=True)
        loss_contrastive = torch.mean((1-label)*torch.pow(euclidean_distance,2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance,min=0.0),2))
        return loss_contrastive
















