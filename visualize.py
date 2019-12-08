import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mplimg

vectors = np.load('test_vectors.npy')
paths = [line.strip() for line in open('data_paths.txt').readlines()]

indices = np.random.permutation(len(paths))[:10]
for i in indices:
	values = []
	for j in range(len(vectors)):
		if i == j: continue
		values.append([np.linalg.norm(vectors[i] - vectors[j]), j])
	values = sorted(values)[:8]
	fig = plt.figure()
	fig.add_subplot(3, 3, 1, label = 'original').set_title('original')
	plt.imshow(mplimg.imread(paths[i]))
	for i in range(8):
		fig.add_subplot(3, 3, i + 2)
		plt.imshow(mplimg.imread(paths[values[i][1]]))
	plt.show()

	