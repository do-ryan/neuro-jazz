# models file

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        # L is the size of the time step dimension of the input data
        super(CNN, self).__init__()

        # self.conv1 = nn.Conv2d().double()
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

        p = 0
        s = 1
        k1 = (12, 96)
        k2 = (12, 96)
        L = (133, 46842)
        num_output_featuremaps = 5

        #self.fc1 = nn.Linear(133*5250, 1).double()
<<<<<<< HEAD
        #self.fc1 = nn.Linear(133*46842, 1).double()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(k1[0], k1[1]), stride=s)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=num_output_featuremaps, kernel_size=(k2[0], k2[1]), stride=s)
        self.fc_inputsize = int((((L[0]-k1[0])/s+1-k2[0])/s+1)*(((L[1]-k1[1])/s+1-k2[1])/s+1)*num_output_featuremaps)
        self.fc1 = nn.Linear(self.fc_inputsize, 2048).double()
        self.fc2 = nn.Linear(2048, 1).double()
=======
        # self.fc1 = nn.Linear(133*29082, 1).double()
        self.fc1 = nn.Linear(62299860, 1).double()
>>>>>>> fd2f309fa5fbaa0af0907bf1b6d5725d9e57716b

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
<<<<<<< HEAD
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.fc_inputsize)
=======
        # x = x.contiguous().view(-1, 133*29082)
>>>>>>> fd2f309fa5fbaa0af0907bf1b6d5725d9e57716b
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = x.squeeze(1)

        return x

class GAN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GAN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 133*output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)     # maybe don't need?
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = x.view(133, self.output_size)
        return x