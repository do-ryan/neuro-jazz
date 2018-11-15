# models file

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> 7a8353031e7d7dfbd428aa6c25dda20bd8847200
        self.conv1 = nn.Conv2d(301, 10, 10).double()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5).double()
        self.fc1 = nn.Linear(20 * 29 * , 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
<<<<<<< HEAD
>>>>>>> 20669f425b781412eadd1744cb3514b71e2fed20

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

        self.fc1 = nn.Linear(133*5484, 1).double()

    def forward(self, x):
<<<<<<< HEAD

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

        x = x.contiguous().view(-1, 133*5484)
        x = F.relu(self.fc1(x))

=======
=======


    def forward(self, x):
>>>>>>> 7a8353031e7d7dfbd428aa6c25dda20bd8847200
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20 * 29 *  )
        x = F.relu(self.fc1(x.float()))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze(1) # Flatten to [batch_size]
<<<<<<< HEAD
>>>>>>> 20669f425b781412eadd1744cb3514b71e2fed20
=======
>>>>>>> 7a8353031e7d7dfbd428aa6c25dda20bd8847200
        return x