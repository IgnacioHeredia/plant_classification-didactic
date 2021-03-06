# -*- coding: utf-8 -*-
"""
Plant Classification webpage

Author: Ignacio Heredia
Date: December 2016

Descrition:
This script launches a basic webpage interface to return results on the plant classification.
To launch the webpage, enter in Ubuntu terminal:
    export FLASK_APP=serve.py
    python -m flask run

Tip:
To host the app in a subpath through a proxy_pass with nginx check Ross's anwer in [1].
Redirections must then be made with either:
* redirect(url_for('intmain', _external=True))    
* redirect('./')

References:
[1] https://stackoverflow.com/questions/25962224/running-a-flask-application-at-a-url-that-is-not-the-domain-root
"""

from flask import Flask, render_template, request, send_from_directory, redirect, url_for, json, Response, make_response
import os
from webpage_utils import url_prediction, localfile_prediction, print_error, label_list_to_html
homedir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


# Configuration parameters of the web application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG'])
if os.path.isfile('secret_key.txt'):
    app.secret_key = open('secret_key.txt', 'r').read()
else:
    app.secret_key = 'devkey, should be in a file'

# Create labels.html from synsets.txt
label_list_to_html(os.path.join(homedir, 'model_files', 'data', 'synsets.txt'))


@app.route('/')
def intmain():
    return render_template("index.html")


@app.route('/labels')
def label_list():
    return render_template('label_list.html')


@app.route('/robots.txt')
def static_from_root():
    return send_from_directory(app.static_folder, request.path[1:])


@app.route('/url_upload', methods=['POST'])
def url_post():
    url_list = request.form['url']
    url_list = [i.replace(' ', '') for i in url_list.split(' ') if i != '']
    
    message = url_prediction(url_list)
    
    if message['status'] == 'error':
        print_error(app, message)
        return redirect(url_for('intmain', _external=True))
    
    if message['status'] == 'OK':
        return render_template('results.html', predictions=message)


@app.route('/local_upload', methods=['POST'])
def local_post():
    uploaded_files = request.files.getlist("local_files")
    
    message = localfile_prediction(app, uploaded_files)
    
    if message['status'] == 'error':
        print_error(app, message)
        return redirect(url_for('intmain', _external=True))
    
    if message['status'] == 'OK':
        return render_template('results.html', predictions=message)


if __name__ == '__main__':
    app.debug = False
    app.run()
