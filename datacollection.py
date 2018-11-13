import os
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from music21 import *


# Authentic Music
page_url = 'https://bushgrafts.com/midi/'
r = requests.get(page_url)
data = r.text
soup = BeautifulSoup(data)

# Authentic music
i = 1
for link in soup.findAll('a', href=True):
    if os.path.splitext(os.path.basename(link['href']))[1] == '.mid':
        f = requests.get(link['href'])
        open('./training_data./authentic./%s.midi'%(i), 'wb').write(f.content)
        i = i+1


# Non-authentic music
page_url = r"https://www.link.cs.cmu.edu/melody-generator/"

# specify the fields that need to be changed, and the values they will be changed to
fields = ['tonality_factor', 'set_mode', 'tonic_endpoints', 'proximity_factor', 'repeated_notes', 'meter_factor',
          'rubato_factor', 'rhythmic_anchoring', 'numbeats']
values = ['0', '-1', '0', '0', '1', '0', '1', '0', '200']

# use selenium to open the page, locate the necessary fields, clear the input, enter new input, and then click submit
download_dir = r"C:\Users\jiyun\PycharmProjects\mie324\NeuroJazz\training_data\nonauthentic"
options = webdriver.ChromeOptions()
options.add_experimental_option('prefs', {"download.default_directory": download_dir})
driver = webdriver.Chrome(r'C:\Users\jiyun\Downloads\chromedriver_win32/chromedriver.exe', chrome_options=options)

for j in range(300):
    i = 0
    driver.get(page_url)
    for i in range(len(fields)):
        sbox = driver.find_element_by_name(fields[i])
        sbox.clear()
        sbox.send_keys(values[i])
        i = i+1
    sbox.submit()
    link = driver.find_elements_by_link_text('MIDI file')[0]
    link.click()
    j = j+1


# Clean authentic data to only be piano music
directory = 'training_data/authentic'
for file in os.listdir(directory):
    data = converter.parse(os.path.join(directory, file))
    s2 = instrument.partitionByInstrument(data)
    temp = stream.Stream()
    if s2 is not None:
        if len(s2) == 1:
            data.write('midi', './training_data./authenticpiano./%s' % (file))
        else:
            for part in s2:
                if isinstance(part.getInstrument(), instrument.Piano) or part == None:
                    temp.append(part)
            temp.write('midi', './training_data./authenticpiano./%s' % (file))
    elif s2 is None:
        data.write('midi', './training_data./authenticpiano./%s' % (file))
    print('%s'%(file))
