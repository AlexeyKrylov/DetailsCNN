from torch.utils.data import Dataset
from PIL import Image
import os


class DetailDataset(Dataset):
    def __init__(self, img_dir, train, transform=None):
        self.listd = []
        if train:
            path = f"{img_dir}/train/"
        else:
            path = f"{img_dir}/test/"
        for i in os.listdir(path=path + "big/"):
            self.listd += [(path + "big/" + i, 1)]
        for i in os.listdir(path=path + "small/"):
            self.listd += [(path + "small/" + i, 0)]
        self.transform = transform

    def __len__(self):
        return len(self.listd)

    def __getitem__(self, idx):
        img_path = self.listd[idx][0]
        with open(img_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        label = self.listd[idx][1]
        if self.transform:
            image = self.transform(image)
        return image, label
