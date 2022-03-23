import numpy as np
import os
import sys
import json
sys.path.append(os.path.dirname(os.path.realpath(__file__))) 
from test_utils import data_augmentation


import theano
import theano.tensor as T
import lasagne

import matplotlib.pylab as plt

theano.config.floatX = 'float32'




homedir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Loading label names and label info files 
synsets = np.genfromtxt(os.path.join(homedir, 'model_files', 'data', 'synsets.txt'), dtype='str', delimiter='/n')
try:
    synsets_info = np.genfromtxt(os.path.join(homedir, 'model_files', 'data', 'info.txt'), dtype='str', delimiter='/n')
except:
    synsets_info = np.array(['']*len(synsets))
assert synsets.shape == synsets_info.shape, """
Your info file should have the same size as the synsets file.
Blank spaces corresponding to labels with no info should be filled with some string (eg '-').
You can also choose to remove the info file."""
    
# Load training info
info_files = os.listdir(os.path.join(homedir, 'model_files', 'training_info'))
info_file_name = [i for i in info_files if i.endswith('.json')][0]
info_file = os.path.join(homedir, 'model_files', 'training_info', info_file_name)
with open(info_file) as datafile:
    train_info = json.load(datafile)
mean_RGB = train_info['augmentation_params']['mean_RGB']
output_dim = train_info['training_params']['output_dim']

# Load net weights 
modelweights = os.path.join(homedir, 'model_files', 'training_weights', 'resnet50_6182classes_100epochs.npz')





input_var = T.tensor4('X', dtype=theano.config.floatX)
from models.resnet50 import build_model
net = build_model(input_var, output_dim)
# Load pretrained weights
with np.load(modelweights) as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(net['prob'], param_values)

"""
['bn2a_branch1',
 'bn2a_branch2a',
 'bn2a_branch2b',
 'bn2a_branch2c',
 'bn2b_branch2a',
 'bn2b_branch2b',
 'bn2b_branch2c',
 'bn2c_branch2a',
 'bn2c_branch2b',
 'bn2c_branch2c',
 'bn3a_branch1',
 'bn3a_branch2a',
 'bn3a_branch2b',
 'bn3a_branch2c',
 'bn3b_branch2a',
 'bn3b_branch2b',
 'bn3b_branch2c',
 'bn3c_branch2a',
 'bn3c_branch2b',
 'bn3c_branch2c',
 'bn3d_branch2a',
 'bn3d_branch2b',
 'bn3d_branch2c',
 'bn4a_branch1',
 'bn4a_branch2a',
 'bn4a_branch2b',
 'bn4a_branch2c',
 'bn4b_branch2a',
 'bn4b_branch2b',
 'bn4b_branch2c',
 'bn4c_branch2a',
 'bn4c_branch2b',
 'bn4c_branch2c',
 'bn4d_branch2a',
 'bn4d_branch2b',
 'bn4d_branch2c',
 'bn4e_branch2a',
 'bn4e_branch2b',
 'bn4e_branch2c',
 'bn4f_branch2a',
 'bn4f_branch2b',
 'bn4f_branch2c',
 'bn5a_branch1',
 'bn5a_branch2a',
 'bn5a_branch2b',
 'bn5a_branch2c',
 'bn5b_branch2a',
 'bn5b_branch2b',
 'bn5b_branch2c',
 'bn5c_branch2a',
 'bn5c_branch2b',
 'bn5c_branch2c',
 'bn_conv1',
 'conv1',
 'conv1_relu',
 'fc1000',
 'input',
 'pool1',
 'pool5',
 'prob',
 'res2a',
 'res2a_branch1',
 'res2a_branch2a',
 'res2a_branch2a_relu',
 'res2a_branch2b',
 'res2a_branch2b_relu',
 'res2a_branch2c',
 'res2a_relu',
 'res2b',
 'res2b_branch2a',
 'res2b_branch2a_relu',
 'res2b_branch2b',
 'res2b_branch2b_relu',
 'res2b_branch2c',
 'res2b_relu',
 'res2c',
 'res2c_branch2a',
 'res2c_branch2a_relu',
 'res2c_branch2b',
 'res2c_branch2b_relu',
 'res2c_branch2c',
 'res2c_relu',
 'res3a',
 'res3a_branch1',
 'res3a_branch2a',
 'res3a_branch2a_relu',
 'res3a_branch2b',
 'res3a_branch2b_relu',
 'res3a_branch2c',
 'res3a_relu',
 'res3b',
 'res3b_branch2a',
 'res3b_branch2a_relu',
 'res3b_branch2b',
 'res3b_branch2b_relu',
 'res3b_branch2c',
 'res3b_relu',
 'res3c',
 'res3c_branch2a',
 'res3c_branch2a_relu',
 'res3c_branch2b',
 'res3c_branch2b_relu',
 'res3c_branch2c',
 'res3c_relu',
 'res3d',
 'res3d_branch2a',
 'res3d_branch2a_relu',
 'res3d_branch2b',
 'res3d_branch2b_relu',
 'res3d_branch2c',
 'res3d_relu',
 'res4a',
 'res4a_branch1',
 'res4a_branch2a',
 'res4a_branch2a_relu',
 'res4a_branch2b',
 'res4a_branch2b_relu',
 'res4a_branch2c',
 'res4a_relu',
 'res4b',
 'res4b_branch2a',
 'res4b_branch2a_relu',
 'res4b_branch2b',
 'res4b_branch2b_relu',
 'res4b_branch2c',
 'res4b_relu',
 'res4c',
 'res4c_branch2a',
 'res4c_branch2a_relu',
 'res4c_branch2b',
 'res4c_branch2b_relu',
 'res4c_branch2c',
 'res4c_relu',
 'res4d',
 'res4d_branch2a',
 'res4d_branch2a_relu',
 'res4d_branch2b',
 'res4d_branch2b_relu',
 'res4d_branch2c',
 'res4d_relu',
 'res4e',
 'res4e_branch2a',
 'res4e_branch2a_relu',
 'res4e_branch2b',
 'res4e_branch2b_relu',
 'res4e_branch2c',
 'res4e_relu',
 'res4f',
 'res4f_branch2a',
 'res4f_branch2a_relu',
 'res4f_branch2b',
 'res4f_branch2b_relu',
 'res4f_branch2c',
 'res4f_relu',
 'res5a',
 'res5a_branch1',
 'res5a_branch2a',
 'res5a_branch2a_relu',
 'res5a_branch2b',
 'res5a_branch2b_relu',
 'res5a_branch2c',
 'res5a_relu',
 'res5b',
 'res5b_branch2a',
 'res5b_branch2a_relu',
 'res5b_branch2b',
 'res5b_branch2b_relu',
 'res5b_branch2c',
 'res5b_relu',
 'res5c',
 'res5c_branch2a',
 'res5c_branch2a_relu',
 'res5c_branch2b',
 'res5c_branch2b_relu',
 'res5c_branch2c',
 'res5c_relu']
"""

#img = data_augmentation(['/home/ignacio/image_recognition/data/demo-images/image1.jpg'])

#layer_list = ['input', 'conv1', 'res2c', 'res3d', 'res4f', 'res5c', 'prob']
#
## Define test function
#test_prediction = lasagne.layers.get_output(net['prob'], deterministic=True)
#test_fn = theano.function([input_var], test_prediction)
#
#
#b = test_fn(img)



layer_list = ['input', 'conv1', 'res2c', 'res3d', 'res4f', 'res5c', 'prob']
img_names = []

# Define test function
test_prediction = lasagne.layers.get_output([net[l] for l in layer_list], deterministic=True)
test_fn = theano.function([input_var], test_prediction)


#b = test_fn(img)
#plt.ioff()
#for layer_name in layer_list[1:-1]:
#    layer_num= layer_list.index(layer_name)
#    img_list = []
#    for i, data in enumerate(b[layer_num][0, :10]):
#        fig, ax = plt.subplots( nrows=1, ncols=1 )
#        ax.imshow(data)
#        ax.get_xaxis().set_visible(False)
#        ax.get_yaxis().set_visible(False)
#        img_name = 'images/{}_{}.jpg'.format(layer_name, i)
#        img_list.append(img_name)
#        print img_name
#        fig.savefig(img_name, bbox_inches='tight', pad_inches=0)