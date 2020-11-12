import torch


class Config(object):
    bottleneck_size = 128
    num_classes = 110475

    path_to_model = 'checkpoints'

    train_batch_size = 8
    test_batch_size = 60

    num_workers = 4

    max_epoch = 50

    train_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
