import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
from PIL import Image
import sys 
sys.path.append(os.path.abspath("/home/tanadun/CXR-Binary-Classifier/test_densenet2.py"))
from test_densenet2 import run_test


app = Flask(__name__)
cors = CORS(app, resorces={r'/*': {"origins": '*'}}) 

def allowed_file_ai(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg',}

@app.route('/run',methods=['POST'])
def runtik():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file_ai(file.filename):
        print('from run()')
        file = request.files['file']
        result = run_test(file)
        print('returned', result)
        return jsonify(result)



@app.route('/', methods=['GET'])
def upload_file():
    return 'Hello'

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded = True, debug = True)