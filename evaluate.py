import os

import torch
from tqdm import tqdm
import numpy as np

from models.model import ReSimpleModel

from config.config import Config

from create_data import read_data

def read_test_data():
    testing_data = read_data()
    testing_image_paths = testing_data['imgpath'].tolist()
    return testing_image_paths


def get_test_data_loader():
    dataset = read_data()
    return torch.utils.data.DataLoader(dataset, batch_size = Config.test_batch_size, shuffle = False, num_workers = Config.num_workers)


def main(model_path, output_file_path):
    model = ReSimpleModel(bottleneck_size=Config.bottleneck_size).to(Config.train_device)
    model.load_state_dict(torch.load(model_path))
    model.train(False)

    img_size = (model.input_height, model.input_width)

    test_data = read_test_data()
    test_dl = get_test_data_loader()

    image_vectors = []
    for data_input, label in tqdm(test_dl):
        data_input = data_input.to(Config.train_device)
        with torch.set_grad_enabled(False):
            feature = model(data_input)
            image_vectors.append(feature.cpu().detach().numpy())

    np.save(output_file_path, np.concatenate(image_vectors, axis=0))


if __name__ == '__main__':
    # this line required for using more than 1 worker in dataloader
    #os.system(f"sudo mount -o remount,size=80G /dev/shm")
	
    main(model_path='checkpoints/general_v7_128_32_7.183936.h5', output_file_path='test_vectors.npy')