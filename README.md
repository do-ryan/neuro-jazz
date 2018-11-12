# NeuroJazz

**Background**

Jazz is a genre of music characterized primarily by improvisation. Improvisation is spontaneous composition, which is a form of creative ideation: a unique capability that humans have over machines. However, there have been a large number of recent attempts to emulate musical composition algorithmically [1], using methods such as: genetic algorithms (i.e. “survival of the fittest” evolution), rule-based algorithms, and most recently, data-driven deep learning. 

A variety of neural net based approaches have been used for generation of music. Most commonly, they involve some variation of a RNN with long-short term memory (LSTM) [2, 3],  training on .MIDI files (symbolic music files). However, recent success was found in an alternative approach using a deep convolutional generative adversarial network (DCGAN) [4, 5]. A DCGAN conventionally works in tandem with one other component: a discriminator CNN, and the overall system operates as follows:

