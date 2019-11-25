import os, numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class lampDataset(Dataset):
	def __init__(self, root_dir, transform = None, mode = 'train'):
		self.transform = transform
		self.mode = mode
		self.names = []
		self.classes = {}
		for cpath, dirs, files in os.walk(root_dir):
			for file in files: self.names.append(os.path.join(cpath, file))
		self.names = np.random.RandomState(seed=42).permutation(self.names)
		if self.mode == 'train':
			self.file_count = int(len(self.names) * 0.8)
		else:
			#assert self.mode == 'test'
			self.names = self.names[::-1]
			self.file_count = len(self.names) - int(len(self.names) * 0.8)
		self.names = self.names[:self.file_count]
	
	def __len__(self):
		return len(self.names)
	
	def __getitem__(self, idx):
		assert idx < self.file_count
		current_name = self.names[idx]
		class_string = current_name.split('\\')[-2]
		if not class_string in self.classes:
			idx = len(self.classes)
			self.classes[class_string] = idx
		sample = {'image': Image.open(current_name).convert('RGB'), 'class': self.classes[class_string] }
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
dataset = lampDataset(root_dir = os.path.join(os.getcwd(), 'dataset'), transform = resizeAndTensor((224, 224)))
for i in range(20): print(dataset.names[i])
"""