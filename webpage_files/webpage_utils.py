# -*- coding: utf-8 -*-
"""
Plant Classification webpage auxiliary functions

Author: Ignacio Heredia
Date: December 2016
"""
import numpy as np
import os, sys
import requests
import json
from flask import flash, Markup
from werkzeug import secure_filename
import random
import StringIO, base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt 

homedir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(homedir)
from model_files.test_utils import load_model, load_saliency_function, data_augmentation, open_image


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
weights_files = os.listdir(os.path.join(homedir, 'model_files', 'training_weights'))
weights_file_name = [i for i in weights_files if i.endswith('.npz')][0]
weights_file_path = os.path.join(homedir, 'model_files', 'training_weights', weights_file_name)
layer_list=['input', 'conv1', 'res2c', 'res3d', 'res4f', 'res5c', 'prob'] #prob should alwys be present (and be present in the last position)
test_func = load_model(weights_file_path, output_dim=output_dim, layer_list=layer_list)
saliency_func = load_saliency_function(weights_file_path, output_dim=output_dim)


def catch_url_error(url_list):
    error_dict = {}
    
    # Error catch: Empty query
    if not url_list:
        error_dict['Error_type'] = 'Empty query'
        return error_dict
           
    for i in url_list:    
        
        # Error catch: Inexistent url        
        try:
            url_type = requests.head(i).headers.get('content-type')
        except:
            error_dict['Error_type'] = 'Failed url connection'
            error_dict['Error_description'] = """Check you wrote the url address correctly."""
            return error_dict
            
        # Error catch: Wrong formatted urls    
        if url_type.split('/')[0] != 'image':
            error_dict['Error_type'] = 'Url image format error'
            error_dict['Error_description'] = """Some urls were not in image format. Check you didn't uploaded a preview of the image rather than the image itself."""
            return error_dict
        
    return error_dict


def allowed_file(app, filename):
    """
    For a given file, return whether it's an allowed type or not
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


def catch_localfile_error(app, local_list):
    error_dict = {}
    
    # Error catch: Empty query
    if not local_list:
        error_dict['Error_type'] = 'Empty query'
        return error_dict
    
    # Error catch: Image format error
    for f in local_list:
        if not allowed_file(app, f.filename):
            error_dict['Error_type'] = 'Local image format error'
            error_dict['Error_description'] = """At least one file is not in a standard image format (jpg|jpeg|png)."""
            return error_dict
        
    return error_dict


def print_error(app, message):
    app.logger.error(message['Error_type'])
    error_message = '<center><b>{}</b></center>'.format(message['Error_type'])
    if 'Error_description' in message.keys():
        error_message += '<br>{}'.format(message['Error_description'])
    flash(Markup(error_message))
    
    
def image_link(pred_lab):
    """
    Return link to Google images
    """
    base_url = 'https://www.google.es/search?'
    links = []
    for i in pred_lab:
        params = {'tbm':'isch','q':i}
        url = base_url + requests.compat.urlencode(params) 
        links.append(url)
    return links


def wikipedia_link(pred_lab):
    """
    Return link to wikipedia webpage
    """
    base_url = 'https://en.wikipedia.org/wiki/'
    links = []
    for i in pred_lab:
        url = base_url + i.replace(' ', '_')
        links.append(url)
    return links
    

def successful_message(pred_lab, pred_prob):
    
    lab_name = synsets[pred_lab].tolist()
    
    message = {'pred_lab': lab_name,   
               'pred_prob':pred_prob.tolist(),
               'google_images_link': image_link(lab_name),
               'wikipedia_link': wikipedia_link(lab_name),
               'info': synsets_info[pred_lab].tolist(),
               'status': 'OK'}
    return message


def fig_encode_64(fig):
    """
    Create base64 encoded image for urls
    """
    imgdata = StringIO.StringIO()
    fig.savefig(imgdata, format='png', bbox_inches='tight', pad_inches=0)
    imgdata.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(imgdata.buf)


def get_net_outputs(im_path, aug_params={}):
    """
    Saves images of intermediate layers and returns image paths and predictions.
    """
    img = data_augmentation([im_path], **aug_params)
    layer_outputs = test_func(img)
    layer_names, layer_imgs_paths = [], []
    filters_per_layer = 12 # show only a certain amount of filter outputs per layer
    plt.ioff()
    for layer_name in layer_list[:-1]:
        layer_num = layer_list.index(layer_name)
        img_list = []
        layer_output = layer_outputs[layer_num]
        for i, data in enumerate(layer_output[0, :filters_per_layer]):
            fig, ax = plt.subplots(1,1)
            ax.imshow(data)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            img_list.append(fig_encode_64(fig))
            plt.close()

        layer_imgs_paths.append(img_list)
        layer_names.append(layer_name + ' ({} feature maps)'.format(layer_output.shape[1]))
        
    layers_dict = {'layer_names': layer_names,
                   'layer_images_path': layer_imgs_paths}
    
    pred_prob = layer_outputs[-1][0]
    args = pred_prob.argsort()[-5:][::-1]  # top5 predicted labels
    pred_lab = args
    return pred_lab, pred_prob[args], layers_dict


def saliency_maps(im_path, aug_params={}, magnitude=False):
    """
    Create sensitivity maps using guided backpropagation
    """
    
    img = data_augmentation([im_path], **aug_params)
    saliency_map, max_class = saliency_func(img) #shape (N,3,224,224)

    if magnitude:
        saliency_map = saliency_map ** 2    
    saliency_map = np.mean(saliency_map, axis=0) #mean across batch (1 image)
    saliency_map = np.abs(saliency_map)
    saliency_map = np.sum(saliency_map, axis=0) #sum accross channels
    
    vmax = np.percentile(saliency_map, 99)
    vmin = np.amin(saliency_map)
    saliency_map = np.clip((saliency_map - vmin)/(vmax - vmin), 0, 1)
    
    plt.ioff()
    map_list = []
    img = open_image(im_path, aug_params['filemode'])
    ori_im = np.array(img.resize((224, 224)))
    
    # Plot original image
    fig, ax = plt.subplots(1,1)
    ax.imshow(ori_im)
    ax.set_title('Original image')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    map_list.append(fig_encode_64(fig))
    
    # Plot saliency map
    fig, ax = plt.subplots(1,1)
    cf = ax.imshow(saliency_map, clim=[np.amin(saliency_map), np.amax(saliency_map)])
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('Saliency map')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    map_list.append(fig_encode_64(fig))
    
    # Plot contour plots of saliency map over the image
    fig, ax = plt.subplots(1,1)
    ax.imshow(ori_im)
    CS = ax.contour(np.arange(saliency_map.shape[0]), np.arange(saliency_map.shape[1]), saliency_map)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title('Contour plot')   
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    map_list.append(fig_encode_64(fig))
    
    plt.close('all')
    
    return {'saliency_maps': map_list}

    
def url_prediction(url_list):
    
    # Catch errors (if any)
    error_message = catch_url_error(url_list)
    if error_message:
        message = {'status': 'error'}
        message.update(error_message)      
        return message
    
    # Predict
    im_path = url_list[0]
    aug_params = {'mean_RGB': mean_RGB, 'filemode':'url', 'mode': 'minimal'}
    pred_lab, pred_prob, layers_dict = get_net_outputs(im_path=im_path, aug_params=aug_params)

    response_dict = layers_dict
    response_dict.update(successful_message(pred_lab, pred_prob))
    
    saliency_map_dict = saliency_maps(im_path=url_list[0], aug_params={'mean_RGB': mean_RGB, 'filemode':'url', 'mode': 'minimal'})
    response_dict.update(saliency_map_dict)
    
    return response_dict


def localfile_prediction(app, uploaded_files):
    
    # Catch errors (if any)
    error_message = catch_localfile_error(app, uploaded_files)
    if error_message:
        message = {'status': 'error'}
        message.update(error_message)      
        return message
    
    # Save images
    filenames = []
    for f in uploaded_files:
#        filename = secure_filename(f.filename)
        filename = str(random.randint(0, 1000000000))
        file_path = os.path.join(homedir, 'webpage_files', 'templates', 'uploads', filename)
        f.save(file_path)
        filenames.append(file_path)
    
    # Predict
    im_path = filenames[0]
    aug_params = {'mean_RGB': mean_RGB, 'filemode':'local', 'mode': 'minimal'}
    pred_lab, pred_prob, layers_dict = get_net_outputs(im_path=im_path, aug_params=aug_params)

    response_dict = layers_dict
    response_dict.update(successful_message(pred_lab, pred_prob))
    response_dict.update(saliency_maps(im_path=im_path, aug_params=aug_params))
    
    # Remove cache images
    for f in filenames:
        os.remove(f)
    
    return response_dict


def label_list_to_html(labels_file):
    """
    Transform the labels_list.txt to an html file to show as database.

    Parameters
    ----------
    labels_file : path to txt file
        Name of labels file (synsets.txt)

    """
    display = """
    <!DOCTYPE html>
    <html lang="en">
    
    <head>

		<!-- Basic Page Needs
		–––––––––––––––––––––––––––––––––––––––––––––––––– -->
		<meta charset="utf-8">
		<title>Deep Learning @ IFCA</title>
		<meta name="description" content="">
		<meta name="author" content="">

		<!-- Mobile Specific Metas
		–––––––––––––––––––––––––––––––––––––––––––––––––– -->
		<meta name="viewport" content="width=device-width, initial-scale=1">

		<!-- FONT
		–––––––––––––––––––––––––––––––––––––––––––––––––– -->
		<link href="//fonts.googleapis.com/css?family=Raleway:400,300,600" rel="stylesheet" type="text/css">

		<!-- CSS
		–––––––––––––––––––––––––––––––––––––––––––––––––– -->
		<link type= "text/css" rel="stylesheet" href="./static/css/normalize.css">
		<link type= "text/css" rel="stylesheet" href="./static/css/skeleton.css">
		<link type= "text/css" rel="stylesheet" href="./static/css/general.css">
		<link type= "text/css" rel="stylesheet" href="./static/css/custom.css">

		<!-- Scripts
		–––––––––––––––––––––––––––––––––––––––––––––––––– -->
		<script src="//ajax.googleapis.com/ajax/libs/jquery/2.1.1/jquery.min.js"></script>
		<script src="https://cdn.rawgit.com/google/code-prettify/master/loader/run_prettify.js"></script>

		<!-- Favicon
		–––––––––––––––––––––––––––––––––––––––––––––––––– -->
		<link rel="icon" type="image/png" href="./static/images/favicon.png">
		<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
            
    </head>
    
    <body>
    
    <!-- Primary Page Layout
    –––––––––––––––––––––––––––––––––––––––––––––––––– -->
    <div class="container">
        <section class="header">
            <h1 class="center"  style="margin-top: 25%">Plant Recognition Demo</h1>
        </section>
        <div class="docs-section">
            <div class="row">
              <div class="eight columns offset-by-two column">
                  
                  <h4>Labels</h4>
                  <p id="show_predictions"></p>
                  <script>document.getElementById("show_predictions").innerHTML = text;</script>
                  <br>"""
    homedir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    labels = np.genfromtxt(os.path.join(homedir, 'model_files', 'data', labels_file), dtype='str', delimiter='/n')
    labels = np.insert(labels, np.arange(len(labels)) + 1, '<br>')
    display += " ".join(labels)                
    display += """ 
              </div>  
          </div>
        </div>
    </div>
    </body>
    </html>"""
  
    with open("templates/label_list.html", "w") as text_file:
        text_file.write(display)
