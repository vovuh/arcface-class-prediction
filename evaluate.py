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


# функция для получения количества классов по заданной характеристике
# (необходима для того, чтобы создать классификатор с такими же параметрами,
# которые были указаны при обучении)
def get_num_classes(property_name):
    # получение датасета
    dataset = lamp_dataset(create_data.read_data(property_name),
                           transform=transforms.Compose([
                               transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                           ]))
    # возвращаемое значение - количество различных классов
    return len(set(dataset.classes))


# считывание путей к изображениям
def read_test_data():
    testing_data = create_data.read_data()
    testing_image_paths = testing_data["imgpath"].tolist()
    return testing_image_paths


# получения DataLoader с заданным transform
def get_test_data_loader(transform):
    dataset = lamp_dataset(create_data.read_data(), transform=transform)
    # единственное, что стоит отметить - здесь отключено перемешивание,
    # так как оно бы мешало сопоставлять вектора и пути к изображениям
    return DataLoader(dataset, batch_size=Config.test_batch_size, shuffle=False, num_workers=Config.num_workers)


# функция для добавления результатов модели к векторам изображений
def epxand_vectors(model, test_dl, image_vectors):
    index = 0
    # перебор данных из DataLoader
    for data_input, label in tqdm(test_dl):
        # отправка данных на устройство
        data_input = data_input.to(Config.train_device)
        # градиенты отключаются, так как это не фаза обучения
        with torch.set_grad_enabled(False):
            # получение векторов для текущего батча
            feature = model(data_input)
            # перенос их на CPU и преобразование в вектора numpy
            numpy_feature = feature.cpu().detach().numpy()
            # запись результатов в соответствующие вектора изображений
            for i in range(len(numpy_feature)):
                image_vectors[index] = np.append(image_vectors[index], numpy_feature[i])
                index += 1
    # возвращение векторов
    return image_vectors


# функция для обработки ArcFace
def load_arcface(model_path):
    # получение DataLoader с трансформацией для ArcFace
    test_dl = get_test_data_loader(transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]))
    # загрузка модели ArcFace с параметрами из файла конфигурации
    model = ReSimpleModel(bottleneck_size=Config.bottleneck_size).to(Config.train_device)
    # загрузка модели из словаря
    model.load_state_dict(torch.load(model_path))
    # указание режима работы модели
    model.train(False)

    # создание файла, содержащего пути к изображениям
    # и запись всех путей в него
    test_data = read_test_data()
    with open("data_paths.txt", "w") as f:
        f.writelines(["%s\n" % item for item in test_data])

    # создание списка пустых векторов для всех изображений
    image_vectors = [np.empty(0) for _ in range(len(test_data))]
    # возвращение списка векторов с добавленными значениями
    # после их обработки ArcFace
    return epxand_vectors(model, test_dl, image_vectors)


# функция для обработки классификатора
def load_classifier(model_path, image_vectors, property_name):
    # получение DataLoader с трансформацией для классификатора
    test_dl = get_test_data_loader(transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    # создание модели с параметрами, указанными при обучени
    # (а также отправка модели на нужное устройство)
    model = models.resnet18(pretrained=True).to(Config.train_device)
    num_ftrs = model.fc.in_features
    # получение количества классов
    classes = get_num_classes(property_name)
    model.fc = nn.Linear(num_ftrs, classes).to(Config.train_device)
    # загрузка модели из словаря
    model.load_state_dict(torch.load(model_path))
    # указание режима работы модели
    model.train(False)

    # возвращение списка векторов с добавленными значениями
    # после их обработки классификатором
    return epxand_vectors(model, test_dl, image_vectors)


if __name__ == "__main__":
    # обработка ArcFace
    vectors = load_arcface(model_path="checkpoints/general_v7_128_36_9.462475.h5")
    # обработка первого классификатора
    vectors = load_classifier(model_path="classifier_models/range_classifier.pth",
                              image_vectors=vectors,
                              property_name="Рекомендуемая площадь освещения")
    # обработка второго классификатора
    vectors = load_classifier(model_path="classifier_models/style_classifier.pth",
                              image_vectors=vectors,
                              property_name="Стиль")
    # сохранение векторов в файл test_vectors.npy
    np.save("test_vectors.npy", vectors)
