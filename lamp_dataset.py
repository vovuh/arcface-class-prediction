import os, numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class lamp_dataset(Dataset):
	def __init__(self, data, transform = None, mode = 'train'):
		self.transform = transform
		self.mode = mode
		self.names = data['imgpath'].tolist()
		self.classes = data['class'].tolist()
		if mode == 'train':
			perm = np.random.permutation(len(self.names))
			newnames = ['' for i in range(len(self.names))]
			newclasses = [-1 for i in range(len(self.classes))]
			for i in range(len(self.names)):
				newnames[i] = self.names[perm[i]]
				newclasses[i] = self.classes[perm[i]]
			self.names = newnames
			self.classes = newclasses
	
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
dataset = lampDataset(root_dir = os.path.join(os.getcwd(), 'dataset'), transform = resizeAndTensor((224, 224)))
for i in range(len(dataset)): dataset[i]
print(len(dataset.classes))
"""