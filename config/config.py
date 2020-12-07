import numpy as np


config = {}
config['lr_stage'] = np.array([30,50,75,100,150,400])
#config['lr'] = [0.001,0.001,0.0001,0.00005,0.00001,0.00001] #adam
config['lr'] = [0.01,0.001,0.0001,0.0001,0.00001,0.00001] #sgd