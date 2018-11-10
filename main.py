#this is the main file 
# edited from Jiyu's computer

'''

Pre processing the data:
    - Download .MIDI files of authentic jazz music, courtesy of https://bushgrafts.com/midi/
    - Convert format to music21 object, 'stream'
    - Label this as 'authentic'
    - Create .MIDI files of non-authentic 'noise' music
    - Convert format to music21 object, 'stream'
    - Label this as 'non-authentic'

'''

import os
import requests
from bs4 import BeautifulSoup


# =================================== DATA COLLECTION =========================================== #
page_url = 'https://bushgrafts.com/midi/'
r = requests.get(page_url)
data = r.text
soup = BeautifulSoup(data)

i = 1
for link in soup.findAll('a', href=True):
    if os.path.splitext(os.path.basename(link['href']))[1] == '.mid':
        f = requests.get(link['href'])
        open('./training_data./%s.midi'%(i), 'wb').write(f.content)
        i = i+1
