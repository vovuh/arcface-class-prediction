import os
from torch.utils.data import DataLoader

import torch
from torch.utils import data
import torch.nn.functional as F
import torchvision
from utils import visualizer, view_model
import torch
import numpy as np
import random
import time
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
import torch.optim.lr_scheduler as scheduler
from sklearn.model_selection import train_test_split

from lamp_dataset import *

from config.config import *

from models.focal_loss import *
from models.metrics import *
from models.resnet import *
from models.newmodel import *

import create_data

def train(device, model, data_loader, optimizer, metric_fc, criterion):
	print("\nTrain\n")
	model.train()
	avg_loss = 0
	count = 0
	for ii, data in enumerate(data_loader):
		data_input, label = data
		data_input = data_input.to(device)
		label = label.to(device).long()
		feature = model.vectorize(data_input)
		output = metric_fc(feature, label)
		loss = criterion(output, label)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		avg_loss += loss.data.item()
		count += 1
		if count % 10 == 0:
			sys.stdout.write("\r{} {} {}".format(count, loss.data.item(), avg_loss / count))
	print('\nDone train {:}\n'.format(avg_loss / count))
	
def validate(device, model, data_loader, metric_fc, criterion):
	model.to(device)
	model.train(False)

	avg_loss = 0
	count = 0

	for data in tqdm(data_loader):
		data_input, label = data
		data_input = data_input.to(device)
		label = label.to(device).long()
		with torch.set_grad_enabled(False):
			feature = model(data_input)
			output = metric_fc(feature, label)
			loss = criterion(output, label)
			avg_loss += loss.data.item()
			count += 1

	return avg_loss / count


if __name__ == '__main__':
	all_data = create_data.read_data()
	train_data, val_data = train_test_split(all_data, train_size=0.8, shuffle=True, random_state=42)
	
	train_dataset = lamp_dataset(train_data, transform = resizeAndTensor((224, 224)))
	val_dataset = lamp_dataset(val_data, transform = resizeAndTensor((224, 224)), mode = 'val')
	
	opt = Config()
	if opt.display: visualizer = Visualizer()
	device = torch.device("cuda")
	
	if opt.loss == 'focal_loss': criterion = FocalLoss(gamma=2)
	else: criterion = torch.nn.CrossEntropyLoss()

	model = ReSimpleModel(bottleneck_size=opt.bottleneck_size)
	model.set_gr(False)
	
	metric_fc = ArcMarginProduct(opt.bottleneck_size, opt.num_classes, s=30, m=0.7, easy_margin=True) # first arg =64 ? 
	
	model.to(device)
	model = DataParallel(model)
	metric_fc.to(device)
	metric_fc = DataParallel(metric_fc)

	if opt.optimizer == 'sgd':
		optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
									lr=opt.lr, weight_decay=opt.weight_decay)
	else:
		optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
									lr=opt.lr, weight_decay=opt.weight_decay)
	
	scheduler = scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.3, verbose=True, threshold=1e-2)

	start = time.time()
	for i in range(opt.max_epoch):
		trainloader = DataLoader(train_dataset, batch_size = opt.train_batch_size, shuffle = True, num_workers = opt.num_workers)
		valloader = DataLoader(val_dataset, batch_size = opt.test_batch_size, shuffle = False, num_workers = opt.num_workers)
		
		train(device, model, trainloader, optimizer, metric_fc, criterion)
		res = validate(device, model, valloader, metric_fc, criterion)
		scheduler.step(res)
		print("Shed step {}".format(res))
		
		model.save(os.path.join(cf.PATH_TO_MODEL, 'general_v7_{:}_{:}_{:2f}.h5'.format(BOTTLENECK_SIZE, epoch_num, res)))
