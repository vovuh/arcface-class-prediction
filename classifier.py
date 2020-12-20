from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
import matplotlib.pyplot as plt
import time
import copy
from torch.utils.data import DataLoader
from lamp_dataset import lamp_dataset
from sklearn.model_selection import train_test_split

import create_data


def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=25):
    # получение времени для вывода статистики
    since = time.time()

    # создание словаря, содержащего лучшую модель
    best_model_wts = copy.deepcopy(model.state_dict())
    # запоминание лучшего значения точности
    best_acc = 0.0

    for epoch in range(num_epochs):
        # вывод информации о текущей эпохе
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # каждая эпоха имеет две фазы:
        # обучение и валидацию
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # установка модели в режим обучения
            else:
                model.eval()   # установка модели в режим валидации

            # текущая функция потерь
            running_loss = 0.0
            # количество корректных классификаций
            running_corrects = 0

            # перебор данных для обучения/валидации
            for inputs, labels in dataloaders[phase]:
                # отправка данных на устройство
                inputs = inputs.to(device)
                labels = labels.to(device)

                # обнуление градиентов
                optimizer.zero_grad()

                # градиенты включатся только при прямом проходе
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # происходит обратный проход + оптимизация,
                    # если текущий режим - это обучение
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # подсчет статистики
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            # функция потерь для эпохи
            epoch_loss = running_loss / dataset_sizes[phase]
            # значение корректности эпохи
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # вывод статистической информации
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # если была фаза валидации
            # и текущая модель стала лучше
            # самой лучшей из уже обработанных
            if phase == 'val' and epoch_acc > best_acc:
                # то эта модель сохраняется
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    # вывод времени, затраченного на обучение
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # вывод лучшей точности модели
    print('Best val Acc: {:4f}'.format(best_acc))

    # загрузка лучшей модели из сохраненной
    # (так как на каждой эпохе словарь состояний
    # переписывается заново)
    model.load_state_dict(best_model_wts)
    # возвращение модели
    return model


if __name__ == "__main__":
    plt.ion()
    # устройство обучения - GPU, если есть cuda, иначе CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # функции для трансформации данных при обучении классификаторов
    data_transforms = {
        'train': transforms.Compose([
            # случайное обрезание изображение до 224х224
            transforms.RandomResizedCrop(224),
            # переворот
            transforms.RandomHorizontalFlip(),
            # превращение в тензор
            transforms.ToTensor(),
            # нормализация цветов
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # изменение размера
            transforms.Resize(256),
            # обрезание по центру
            transforms.CenterCrop(224),
            # превращение в тензор
            transforms.ToTensor(),
            # нормализация цветов
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # перебор двух свойств:
    # Стиль
    # и
    # Рекомендуемая площадь освещения
    for property_name in ["Стиль", "Рекомендуемая площадь освещения"]:
        # загрузка данных с указанием property_name
        all_data = create_data.read_data(property_name)
        # разделение на тренировочный датасет и датасет для валидации
        train_data, val_data = train_test_split(all_data, train_size=0.8, shuffle=True, random_state=42)
        # создание самих датасетов
        train_dataset = lamp_dataset(train_data, transform=data_transforms["train"])
        val_dataset = lamp_dataset(val_data, transform=data_transforms["val"], mode="val")
        # создание DataLoader для получившихся датасетов
        image_datasets = {"train": train_dataset, "val": val_dataset}
        # размер батча = 4, включение перемешивания и указание количества потоков
        # равным нулю, потому что Windows сам решает, во сколько потоков обрабатывать батчи
        data_loaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0)
                        for x in ["train", "val"]}
        # получение размеров датасетов
        dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
        # получение списка классов для тренировки
        class_names = image_datasets["train"].classes

        # загрузка претренированной модели resnet18
        model_ft = models.resnet18(pretrained=True)
        # указание количество входных параметров
        num_ftrs = model_ft.fc.in_features

        # указание количества выходных параметров
        # (это значение равно количеству различных классов)
        model_ft.fc = nn.Linear(num_ftrs, len(set(class_names)))

        # отправка модели на необходимое устройство
        model_ft = model_ft.to(device)

        # создание критерия оценки
        criterion = nn.CrossEntropyLoss()

        # создание оптимизатора
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # создание объекта, отвечающего за изменение
        # "обучаемости" в зависимости от количества эпох
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        # тренировка и получение лучшей модели
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, data_loaders, device)
        # сохранение модели в директорию classifier_models/ с названием,
        # соответствующим значению характеристики
        torch.save(
            model_ft.state_dict(),
            "classifier_models/" + ("style_classifier.pth" if property_name == "Стиль" else "range_classifier.pth")
        )
