# models file

import torch 
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, batch_size):
        # L is the size of the time step dimension of the input data
        super(CNN, self).__init__()

        #p = 0 padding
        s = 1
        k1 = (12, 96)
        k2 = (2, 2)
        #L = (133,26502)
        num_output_featuremaps = 5

        #self.fc1 = nn.Linear(133*46842, 1).double()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(k1[0],k1[1]), stride=(12, 48)).double()
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=num_output_featuremaps, kernel_size=(k2[0],k2[1]), stride=k2).double()
        #self.fc_inputsize = int((((L[0]-k1[0])/s+1-k2[0])/s+1)*(((L[1]-k1[1])/s+1-k2[1])/s+1)*num_output_featuremaps)
        self.fc_inputsize = int(280/batch_size) # size of fully connected input
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(self.fc_inputsize, 2000).double()
        self.fc2 = nn.Linear(2000, 500).double()
        self.fc3 = nn.Linear(500, 1).double()

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        #x = x.permute(0, 1, 2, 3)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = x.squeeze()
        x = x.view(-1, self.fc_inputsize)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = x.squeeze(1)
        return x

class GAN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, pitch_range):
        super(GAN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.pitch_range = pitch_range
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, self.pitch_range*output_size) #48 is the number of possible pitches

    def forward(self, x):
        x = x.view(-1, self.input_size)     # maybe don't need?
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(self.batch_size, self.pitch_range, self.output_size)
        return x

