import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    vectors = np.load("test_vectors.npy")
    paths = [line.strip() for line in open("data_paths.txt").readlines()]

    indices = np.random.permutation(len(paths))[:5]
    for i in indices:
        values = []
        for j in range(len(vectors)):
            if i == j:
                continue
            values.append([np.linalg.norm(vectors[i] - vectors[j]), j])
        values = sorted(values)[:8]
        fig = plt.figure()
        fig.add_subplot(3, 3, 1, label="original").set_title("original")
        plt.imshow(mplimg.imread(paths[i]))
        for j in range(8):
            fig.add_subplot(3, 3, j + 2)
            plt.imshow(mplimg.imread(paths[values[j][1]]))
        plt.show()
