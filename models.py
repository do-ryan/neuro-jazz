# models file

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from music21 import converter, instrument, midi, pitch, note, chord

# song = midi.MidiFile.open(filename='training_data/thejazzpage/caravan.midi')
data = converter.parse('training_data/thejazzpage/caravan.midi')
parts = instrument.partitionByInstrument(data)
mf = midi.MidiFile()
mf.open('./training_data./thejazzpage./caravan.midi')
mf.read()
base_midi = midi.translate.midiFileToStream(mf)



print(3)