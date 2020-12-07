import numpy as np


config = {}

config['lr_stage'] = np.array([30,150,180,200,500,10000])
config['lr'] = np.array([0.001,0.0001,0.00001,0.00001,0.0001,0.0001]) #dense net adam

#
# config['lr_stage'] = np.array([30,60,90,120,150,180]) # se lr
# config['lr'] = np.array([0.1,0.01,0.001,0.0001,0.00001,0.00001]) #se net sgd