import os
import pandas as pd

def read_data():
	path = os.path.join(os.getcwd(), 'dataset')
	classes = {}
	for cpath, dirs, files in os.walk(path):
		for file in files:
			file_class = cpath.split('\\')[-1]
			if not file_class in classes:
				idx = len(classes)
				classes[file_class] = idx
	names = []
	labels = []
	for cpath, dirs, files in os.walk(path):
		for file in files:
			names.append(os.path.join(cpath, file))
			labels.append(classes[cpath.split('\\')[-1]])
	data = { 'imgpath' : names, 'class' : labels }
	return pd.DataFrame(data)
