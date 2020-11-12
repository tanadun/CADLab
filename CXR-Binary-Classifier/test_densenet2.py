"""
Yuxing Tang
Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
National Institutes of Health Clinical Center
March 2020
THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import print_function
from __future__ import division
import os
import argparse
import distutils.util
import numpy as np
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
# from sklearn.metrics import roc_auc_score
# from PIL import Image
# import time

import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func
import time
import copy
# from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn import metrics
from CXR_Data_Generator2 import DataGenerator

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch NIH-CXR Testing')

parser.add_argument('--arch', '-a', metavar='ARCH', default='densenet121')
parser.add_argument('--img_size', '-sz', default=256, type=int)
parser.add_argument('--crop_size', '-cs', default=224, type=int)
parser.add_argument('--epoch', '-ep', default=50, type=int)
parser.add_argument('--batch_size', '-bs', default=64, type=int)
parser.add_argument('--learning_rate', '-lr', default=0.001, type=float)
parser.add_argument('--gpu_id', '-gpu', default=0, type=int)
parser.add_argument('--test_labels', default='att', type=str) #default: '': attending radiologist labels. 'con' for radiologist consensus.

def run_test(uploaded_file):
	global args
	args = parser.parse_args()

	model = models.__dict__[args.arch](pretrained=True)

	torch.cuda.set_device(args.gpu_id)

	# number of classes
	numClass = 1

	# modify the last FC layer to number of classes
	num_ftrs = model.classifier.in_features
	model.classifier = nn.Linear(num_ftrs, numClass)

	model = model.cuda()

	model_path = './trained_models_nih/'+args.arch+'_'+str(args.img_size)+\
	'_'+str(args.batch_size)+'_'+str(args.learning_rate)
	print('model_path', model_path)

	model.load_state_dict(torch.load(model_path)['state_dict'])

	return test(uploaded_file, model, batch_size=args.batch_size, \
		img_size=args.img_size, crop_size=args.crop_size, gpu_id=args.gpu_id)


def test(uploaded_file, model, batch_size, img_size, crop_size, gpu_id):

	# -------------------- SETTINGS: CXR DATA TRANSFORMS -------------------
	normalizer = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
	transform = transforms.Compose([
		transforms.Resize(img_size),
		# transforms.RandomResizedCrop(crop_size),
		transforms.CenterCrop(crop_size),
		transforms.ToTensor(),
		transforms.Normalize(normalizer[0], normalizer[1])])
	
	# -------------------- SETTINGS: DATASET BUILDERS -------------------
	datasetTest = DataGenerator(uploaded_file=uploaded_file, transform=transform)
	dataloader = DataLoader(dataset=datasetTest, batch_size=1,
								shuffle=False, num_workers=32, pin_memory=True)
 
	# -------------------- TESTING -------------------
	model.eval()
	count = 0

	with torch.no_grad():
		# Iterate over data.
		for data in dataloader:
			inputs, img_names = data

			count = count+1
			print(count,inputs)

			# wrap them in Variable
			inputs = inputs.cuda(gpu_id, non_blocking=True)
			# forward
			outputs = model(inputs)
			# _, preds = torch.max(outputs.data, 1)
			score = torch.sigmoid(outputs)
			score_np = score.data.cpu().numpy()

			outputs = outputs.data.cpu().numpy()
			result = {
				"filename": str(img_names[0]),
				"score": str(score_np[0]),
				"Diagnostic Result 0":"Low score, nearly 0 = Potential to be Normal",
				"Diagnostic Result 1":"High score, nearly 1 = Potential to be Abnormal"
			}
			#result = str(img_names[0])  + ': ' + str(score_np[0])
			print('img_names[0]',img_names[0])
			print('score_np[0]',score_np[0])
			print('result',result)
			return result
