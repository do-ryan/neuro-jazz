# models file

import torch 
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, batch_size):
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
        #p = 0 padding
        s = 1
        k1 = (12, 12)
        k2 = (4, 4)
        #L = (133,26502)
        num_output_featuremaps = 5

        #self.fc1 = nn.Linear(133*46842, 1).double()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(k1[0],k1[1]), stride=s).double()
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=num_output_featuremaps, kernel_size=(k2[0],k2[1]), stride=s).double()
        #self.fc_inputsize = int((((L[0]-k1[0])/s+1-k2[0])/s+1)*(((L[1]-k1[1])/s+1-k2[1])/s+1)*num_output_featuremaps)
        self.fc_inputsize = int(53120/(batch_size*batch_size*2))
        self.pool = nn.MaxPool2d(3,3)
        self.fc1 = nn.Linear(self.fc_inputsize, 2000).double()
        self.fc2 = nn.Linear(2000, 1).double()

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
        print(x.shape)
        x = torch.unsqueeze(x, dim=1)
        x = x.permute(0, 1, 3, 2)
        x = self.conv1(x) #RuntimeError: Expected 4-dimensional input for 4-dimensional weight [5, 1, 12, 96], but got input of size [10, 18726, 133] instead
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
	x = x.squeeze()
        print(x.shape)
        x = x.view(-1, self.fc_inputsize)
        # x = x.contiguous().view(-1, 133*29082)
        x = self.fc1(x)
        x = F.relu(x)
        x = torch.sigmoid(self.fc2(x))
        x = x.squeeze(1)
        print(x.shape)
        return x

class GAN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(GAN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 36*output_size) #36 is the number of possible pitches

    def forward(self, x):
        x = x.view(-1, self.input_size)     # maybe don't need?
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = x.view(self.batch_size, 36, self.output_size)
        return x

