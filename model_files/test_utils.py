# -*- coding: utf-8 -*-
"""
Miscellanous functions used to evaluate image classification in the demo webpage.

Author: Ignacio Heredia
Date: December 2016
"""

import numpy as np
import os
import sys
from PIL import Image, ImageEnhance
import requests
from io import BytesIO
sys.path.append(os.path.dirname(os.path.realpath(__file__))) 

import theano
import theano.tensor as T
import theano.gpuarray as gpuarray
import lasagne

theano.config.floatX = 'float32'


def load_model(modelweights, output_dim, layer_list=['input', 'conv1', 'res2c', 'res3d', 'res4f', 'res5c', 'prob']):
    """
    Loads a model with some trained weights and returns the test function that
    gives the deterministic predictions.

    Parameters
    ----------
    modelweights : str
        Name of the weights file
    outputdim : int
        Number of classes to predict

    Returns
    -------
    Test function
    """
    print 'Loading the model...'
    input_var = T.tensor4('X', dtype=theano.config.floatX)
    from models.resnet50 import build_model
    net = build_model(input_var, output_dim)
    # Load pretrained weights
    with np.load(modelweights) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(net['prob'], param_values)
    # Define test function
    test_prediction = lasagne.layers.get_output([net[l] for l in layer_list], deterministic=True)
    test_fn = theano.function([input_var], test_prediction)
    return test_fn


def guided_backprop(net):
    """
    Modify the gradient of the relu to implement guided backpropagation
    """
    
    class ModifiedBackprop(object):
    
        def __init__(self, nonlinearity):
            self.nonlinearity = nonlinearity
            self.ops = {}  # memoizes an OpFromGraph instance per tensor type
    
        def __call__(self, x):
            # OpFromGraph is oblique to Theano optimizations, so we need to move
            # things to GPU ourselves if needed.
            if gpuarray.pygpu_activated:
                ctx = theano.gpuarray.basic_ops.infer_context_name(x)
                x = theano.gpuarray.as_gpuarray_variable(x, ctx)
            # We note the tensor type of the input variable to the nonlinearity
            # (mainly dimensionality and dtype); we need to create a fitting Op.
            tensor_type = x.type
            # If we did not create a suitable Op yet, this is the time to do so.
            if tensor_type not in self.ops:
                # For the graph, we create an input variable of the correct type:
                inp = tensor_type()
                # We pass it through the nonlinearity (and move to GPU if needed).
                outp = self.nonlinearity(inp)
                if gpuarray.pygpu_activated:
                    ctx = theano.gpuarray.basic_ops.infer_context_name(outp)
                    outp = theano.gpuarray.as_gpuarray_variable(outp, ctx)
                # Then we fix the forward expression...
                op = theano.OpFromGraph([inp], [outp])
                # ...and replace the gradient with our own (defined in a subclass).
                op.grad = self.grad
                # Finally, we memoize the new Op
                self.ops[tensor_type] = op
            # And apply the memoized Op to the input we got.
            return self.ops[tensor_type](x)
    
    class GuidedBackprop(ModifiedBackprop):
        
        def grad(self, inputs, out_grads):
            (inp,) = inputs
            (grd,) = out_grads
            dtype = inp.dtype
            return (grd * (inp > 0).astype(dtype) * (grd > 0).astype(dtype),)
    
    relu = lasagne.nonlinearities.rectify
    relu_layers = [layer for layer in lasagne.layers.get_all_layers(net['prob'])
                   if getattr(layer, 'nonlinearity', None) is relu]
    modded_relu = GuidedBackprop(relu)  # important: only instantiate this once!
    for layer in relu_layers:
        layer.nonlinearity = modded_relu

    return net


def load_saliency_function(modelweights, output_dim, use_guided_backprop=True):
    """
    Loads a model with some trained weights and returns a function to compute 
    the saliency maps and predicted classes.

    Parameters
    ----------
    modelweights : str
        Name of the weights file
    outputdim : int
        Number of classes to predict

    Returns
    -------
    Test function
    """
    print 'Loading the model...'
    input_var = T.tensor4('X', dtype=theano.config.floatX)
    from models.resnet50 import build_model
    net = build_model(input_var, output_dim)
    # Load pretrained weights
    with np.load(modelweights) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(net['prob'], param_values)
    
    if use_guided_backprop:
        net = guided_backprop(net)

    inp = net['input'].input_var
    outp = lasagne.layers.get_output(net['fc1000'], deterministic=True)
    max_outp = T.max(outp, axis=1) #we select the first prediction
    saliency = theano.grad(max_outp.sum(), wrt=inp) #we sum because cost must be scalar
    max_class = T.argmax(outp, axis=1)
    return theano.function([inp], [saliency, max_class])      


def open_image(filename, filemode):
    if filemode == 'local':
        im = Image.open(filename)
        im = im.convert('RGB')
    elif filemode == 'url':
        filename = BytesIO(requests.get(filename).content)
        im = Image.open(filename)
        im = im.convert('RGB')
    return im

def data_augmentation(im_list, mode='standard', tags=None, params=None, im_size=224,
                      filemode='local', mean_RGB=None):
    """
    Perform data augmentation on some image list using PIL.

    Parameters
    ----------
    im_list : array of strings
        Array where the first column is image_path (or image_url). Optionally
        a second column can be the tags of the image.
        Shape (N,) or (N,2)
    tags : array of strings, None
        If existing, you can the manually modify the data_augmentation function
        (by adding an additional condition to the if, like tags[i]=='fruit')
        to choose which transformations are to be performed to each tag.
    params : dict or None
        Mandatory keys:
        - mirror (bool): allow 50% random mirroring.
        - rotation (bool): allow rotation of [0, 90, 180, 270] degrees.
        - stretch ([0,1] float): randomly stretch image.
        - crop ([0,1] float): randomly take an image crop.
        - zoom ([0,1] float): random zoom applied to crop_size.
          --> Therefore the effective crop size at each iteration will be a 
              random number between 1 and crop*(1-zoom). For example:
                  * crop=1, zoom=0: no crop of the image
                  * crop=1, zoom=0.1: random crop of random size between 100% image and 90% of the image
                  * crop=0.9, zoom=0.1: random crop of random size between 90% image and 80% of the image
                  * crop=0.9, zoom=0: random crop of always 90% of the image
                  Image size refers to the size of the shortest side.
        - pixel_noise (bool): allow different pixel transformations like gaussian noise,
          brightness, color jittering, contrast and sharpness modification.
    mode : {'standard', 'minimal', 'test', None}
        We overwrite the params dict with some defaults augmentation parameters
        - 'minimal': no data augmentation, just resizing
        - 'standard': tipical parameters for data augmentation during training
        - 'test': minimized data augmentation for validation/testing
        - None: we do not overwrite the params dict variable
    im_size : int
        Final image size to feed the net's input (eg. 224 for Resnet).
    filemode : {'local','url'}
        - 'local': filename is absolute path in local disk.
        - 'url': filename is internet url.
    mean_RGB : array, None
        Mean RGB values for your dataset. If not provided, we use some default values.

    Returns
    -------
    Array of shape (N,3,im_size,im_size) containing the augmented images.

    """
    if mean_RGB is None:
        mean_RGB = np.array([107.59348955,  112.1047813,   80.9982362])
    else:
        mean_RGB = np.array(mean_RGB)   

    if mode == 'minimal':
        params = {'mirror':False, 'rotation':False, 'stretch':False, 'crop':False, 'pixel_noise':False}
    if mode == 'standard':
        params = {'mirror':True, 'rotation':True, 'stretch':0.3, 'crop':1., 'zoom':0.3, 'pixel_noise':False}
    if mode == 'test':
        params = {'mirror':True, 'rotation':True, 'stretch':0.1, 'crop':.9, 'zoom':0.1, 'pixel_noise':False}
    
    batch = []
    for i, filename in enumerate(im_list):
        
        if filemode == 'local':
            im = Image.open(filename)
            im = im.convert('RGB')
        elif filemode == 'url':
            filename = BytesIO(requests.get(filename).content)
            im = Image.open(filename)
            im = im.convert('RGB')
                
        if params['stretch']:
            stretch = params['stretch']
            stretch_factor = np.random.uniform(low=1.-stretch/2, high=1.+stretch/2, size=2)
            im = im.resize((im.size * stretch_factor).astype(int))
            
        if params['crop']:
            effective_zoom = np.random.rand() * params['zoom']
            crop = params['crop'] - effective_zoom
            
            ly, lx = im.size
            crop_size = crop * min([ly, lx]) 
            rand_x = np.random.randint(low=0, high=lx-crop_size + 1)
            rand_y = np.random.randint(low=0, high=ly-crop_size + 1)
                
            min_yx = np.array([rand_y, rand_x])
            max_yx = min_yx + crop_size #square crop
            im = im.crop(np.concatenate((min_yx, max_yx)))
            
        if params['mirror']:
            if np.random.random() > 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.random() > 0.5:
                im = im.transpose(Image.FLIP_TOP_BOTTOM)
        
        if params['rotation']:
            rot = np.random.choice([0, 90, 180, 270])
            if rot == 90:
                im = im.transpose(Image.ROTATE_90)
            if rot == 180:
                im = im.transpose(Image.ROTATE_180)
            if rot == 270:
                im = im.transpose(Image.ROTATE_270)            

        if params['pixel_noise']:
            
            #not used by defaul as it does not seem to improve much the performance,
            #but more than DOUBLES the data augmentation processing time.
            
            # Color
            color_factor = np.random.normal(1, 0.3)  #1: original
            color_factor = np.clip(color_factor, 0., 2.)
            im = ImageEnhance.Color(im).enhance(color_factor)
            
            # Brightness
            brightness_factor = np.random.normal(1, 0.2)  #1: original
            brightness_factor = np.clip(brightness_factor, 0.5, 1.5)
            im = ImageEnhance.Brightness(im).enhance(brightness_factor)
            
            # Contrast
            contrast_factor = np.random.normal(1, 0.2)  #1: original
            contrast_factor = np.clip(contrast_factor, 0.5, 1.5)
            im = ImageEnhance.Contrast(im).enhance(contrast_factor)
            
            # Sharpness
            sharpness_factor = np.random.normal(1, 0.4)  #1: original
            sharpness_factor = np.clip(sharpness_factor, 0., 1.)
            im = ImageEnhance.Sharpness(im).enhance(sharpness_factor)

#            # Gaussian Noise #severely deteriorates learning 
#            im = np.array(im)
#            noise = np.random.normal(0, 15, im.shape)
#            noisy_image = np.clip(im + noise, 0, 255).astype(np.uint8)
#            im = Image.fromarray(noisy_image)

        im = im.resize((im_size, im_size))
        batch.append(np.array(im))  # shape (N, 224, 224, 3)

    batch = np.array(batch) - mean_RGB[None, None, None, :]  # mean centering
    batch = batch.transpose(0, 3, 1, 2)  # shape(N, 3, 224, 224)
    batch = batch[:, ::-1, :, :]  # switch from RGB to BGR
    return batch.astype(np.float32)

