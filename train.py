import os
import sys

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

import create_data
from config.config import Config
from lamp_dataset import lamp_dataset, resizeAndTensor
from models.metrics import ArcMarginProduct
from models.model import ReSimpleModel


def train(device, model, data_loader, optimizer, metric_fc, criterion):
    print('\nTrain\n')
    model.train()
    avg_loss = 0
    count = 0
    for ii, data in enumerate(data_loader):
        data_input, label = data
        data_input = data_input.to(device)
        label = label.to(device).long()
        feature = model.vectorize(data_input)
        output = metric_fc(feature, label)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.data.item()
        count += 1
        if count % 10 == 0:
            sys.stdout.write('\r{} {} {}'.format(count, loss.data.item(), avg_loss / count))
    print('\nDone train {:}\n'.format(avg_loss / count))


def validate(device, model, data_loader, metric_fc, criterion):
    model.to(device)
    model.train(False)

    avg_loss = 0
    count = 0

    for data in tqdm(data_loader):
        data_input, label = data
        data_input = data_input.to(device)
        label = label.to(device).long()
        with torch.set_grad_enabled(False):
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            avg_loss += loss.data.item()
            count += 1

    return avg_loss / count


if __name__ == '__main__':
    all_data = create_data.read_data()
    train_data, val_data = train_test_split(all_data, train_size=0.8, shuffle=True, random_state=42)

    train_dataset = lamp_dataset(train_data, transform=resizeAndTensor((224, 224)))
    val_dataset = lamp_dataset(val_data, transform=resizeAndTensor((224, 224)), mode='val')

    opt = Config()
    device = opt.train_device

    criterion = torch.nn.CrossEntropyLoss()

    model = ReSimpleModel(bottleneck_size=opt.bottleneck_size)
    model.set_gr(False)

    metric_fc = ArcMarginProduct(opt.bottleneck_size, opt.num_classes, s=30, m=0.7, easy_margin=True)

    model.to(device)
    metric_fc.to(device)

    optimizer = optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                           lr=0.01, weight_decay=5e-4)
    sched = scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.3, verbose=True, threshold=1e-2)

    for epoch_num in range(opt.max_epoch):
        if epoch_num == 3:
            model.set_gr(True)
            optimizer = optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=0.01,
                                   weight_decay=5e-4)
            sched = scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.3, verbose=True, threshold=1e-2)

        trainloader = DataLoader(train_dataset, batch_size=opt.train_batch_size, shuffle=True,
                                 num_workers=opt.num_workers)
        valloader = DataLoader(val_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=opt.num_workers)

        train(device, model, trainloader, optimizer, metric_fc, criterion)
        res = validate(device, model, valloader, metric_fc, criterion)
        sched.step(res)
        print('Shed step {}'.format(res))

        model.save(
            os.path.join(opt.path_to_model, 'general_v7_{:}_{:}_{:2f}.h5'.format(opt.bottleneck_size, epoch_num, res)))
