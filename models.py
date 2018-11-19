# models file

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # self.conv1 = nn.Conv2dd().double()
        # self.pool = nn.MaxPool2d(2)
        # self.conv2 = nn.Conv2d().double()
        # self.conv3 = nn.Conv2d().double()
        # self.fc1 = nn.Linear()
        # self.fc2 = nn.Linear()
        # self.fc3 = nn.Linear()
        # self.bn1 = nn.BatchNorm2d().double()
        # self.bn2 = nn.BatchNorm2d().double()
        # self.bn3 = nn.BatchNorm2d().double()
        '''
        Assignment 3 code above
        '''

        #self.fc1 = nn.Linear(133*5250, 1).double()
        self.fc1 = nn.Linear(133*29082, 1).double()

    def forward(self, x):

        # x = self.pool(self.bn1(F.relu(self.conv1(x))))
        # x = self.pool(self.bn2(F.relu(self.conv2(x))))
        # x = self.pool(self.bn3(F.relu(self.conv3(x))))
        # x = x.view()
        # x = F.relu(self.fc1(x.float()))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # x = x.squeeze(1) # Flatten to [batch_size]
        '''
        Assignment 3 code above
        '''

        #x = x.contiguous().view(-1, 5250*133)
        x = x.contiguous().view(-1, 133*29082)
        x = F.relu(self.fc1(x))
        x = x.squeeze(1)

        return x
