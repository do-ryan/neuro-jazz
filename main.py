# Main file

from datavisualization import list_instruments
from music21 import *
import os

instruments = []
directory = 'training_data/authentic'
for file in os.listdir(directory):
    data = converter.parse(os.path.join(directory, file))
    instruments.append(list_instruments(data))

print(3)

def main:


if __name__ == "__main__":
    main()