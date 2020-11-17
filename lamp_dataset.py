from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class lamp_dataset(Dataset):
    def __init__(self, data, transform=None, mode="train"):
        self.transform = transform
        self.mode = mode
        self.names = data["imgpath"].tolist()
        self.classes = data["class"].tolist()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img, label = Image.open(self.names[idx]).convert("RGB"), self.classes[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class resizeAndTensor(object):
    def __init__(self, output_size):
        self.my_transforms = transforms.Compose([
            transforms.Resize(output_size),
            transforms.ToTensor(),
        ])

    def __call__(self, sample):
        return self.my_transforms(sample)
