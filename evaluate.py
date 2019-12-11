import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

from models.model import ReSimpleModel

from config.config import Config

from lamp_dataset import lamp_dataset, resizeAndTensor

import create_data

def read_test_data():
    testing_data = create_data.read_data()
    testing_image_paths = testing_data['imgpath'].tolist()
    return testing_image_paths


def get_test_data_loader():
	dataset = lamp_dataset(create_data.read_data(), transform = resizeAndTensor((224, 224)))
	return DataLoader(dataset, batch_size = Config.test_batch_size, shuffle = False, num_workers = Config.num_workers)

    
def main(model_path, output_file_path):
    model = ReSimpleModel(bottleneck_size=Config.bottleneck_size).to(Config.train_device)
    model.load_state_dict(torch.load(model_path))
    model.train(False)

    #img_size = (model.input_height, model.input_width)

    test_data = read_test_data()
    with open('data_paths.txt', 'w') as f:
    	f.writelines(['%s\n' % item for item in test_data])
    
    test_dl = get_test_data_loader()

    image_vectors = []
    for data_input, label in tqdm(test_dl):
        data_input = data_input.to(Config.train_device)
        with torch.set_grad_enabled(False):
            feature = model(data_input)
            image_vectors.append(feature.cpu().detach().numpy())

    np.save(output_file_path, np.concatenate(image_vectors, axis=0))


if __name__ == '__main__':
    main(model_path='checkpoints/general_v7_128_32_7.183936.h5', output_file_path='test_vectors.npy')
