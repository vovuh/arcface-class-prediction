import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import create_data
from config.config import Config
from lamp_dataset import lamp_dataset
from models.model import ReSimpleModel
from torchvision import models, transforms
import torch.nn as nn


def get_num_classes(property_name):
    dataset = lamp_dataset(create_data.read_data(property_name),
                           transform=transforms.Compose([
                               transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                           ]))
    return len(set(dataset.classes))


def read_test_data():
    testing_data = create_data.read_data()
    testing_image_paths = testing_data["imgpath"].tolist()
    return testing_image_paths


def get_test_data_loader(transform):
    dataset = lamp_dataset(create_data.read_data(), transform=transform)
    return DataLoader(dataset, batch_size=Config.test_batch_size, shuffle=False, num_workers=Config.num_workers)


def epxand_vectors(model, test_dl, image_vectors):
    index = 0
    for data_input, label in tqdm(test_dl):
        data_input = data_input.to(Config.train_device)
        with torch.set_grad_enabled(False):
            feature = model(data_input)
            numpy_feature = feature.cpu().detach().numpy()
            for i in range(len(numpy_feature)):
                image_vectors[index] = np.append(image_vectors[index], numpy_feature[i])
                index += 1
    return image_vectors


def load_arcface(model_path):
    test_dl = get_test_data_loader(transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]))
    model = ReSimpleModel(bottleneck_size=Config.bottleneck_size).to(Config.train_device)
    model.load_state_dict(torch.load(model_path))
    model.train(False)

    test_data = read_test_data()
    with open("data_paths.txt", "w") as f:
        f.writelines(["%s\n" % item for item in test_data])

    image_vectors = [np.empty(0) for _ in range(len(test_data))]
    return epxand_vectors(model, test_dl, image_vectors)


def load_classifier(model_path, image_vectors, property_name):
    test_dl = get_test_data_loader(transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    model = models.resnet18(pretrained=True).to(Config.train_device)
    num_ftrs = model.fc.in_features
    classes = get_num_classes(property_name)
    model.fc = nn.Linear(num_ftrs, classes).to(Config.train_device)
    model.load_state_dict(torch.load(model_path))
    model.train(False)

    return epxand_vectors(model, test_dl, image_vectors)


if __name__ == "__main__":
    vectors = load_arcface(model_path="checkpoints/general_v7_128_36_9.462475.h5")
    vectors = load_classifier(model_path="classifier_models/range_classifier.pth",
                              image_vectors=vectors,
                              property_name="Рекомендуемая площадь освещения")
    vectors = load_classifier(model_path="classifier_models/style_classifier.pth",
                              image_vectors=vectors,
                              property_name="Стиль")
    np.save("test_vectors.npy", vectors)
