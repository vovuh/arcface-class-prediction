import os, numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import create_data

class lamp_dataset(Dataset):
	def __init__(self, data, transform = None, mode = 'train'):
		self.transform = transform
		self.mode = mode
		self.names = data['imgpath'].tolist()
		self.classes = data['class'].tolist()
	
	def __len__(self):
		return len(self.names)
	
	def __getitem__(self, idx):
		assert idx < len(self.names)
		sample = {'image': Image.open(self.names[idx]).convert('RGB'), 'class': self.classes[idx]}
		if self.transform: sample = self.transform(sample)
		return sample

class resizeAndTensor(object):
	def __init__(self, output_size):
		self.my_transforms = transforms.Compose([
			transforms.Resize(output_size),
			transforms.ToTensor(),
		])

	def __call__(self, sample):
		img = self.my_transforms(sample['image'])
		return {'image': img, 'class': sample['class']}

"""
dataset = lamp_dataset(create_data.read_data(), transform = resizeAndTensor((224, 224)))
"""