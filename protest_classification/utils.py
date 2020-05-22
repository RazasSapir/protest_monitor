import os

from PIL import Image
import torchvision.models as models
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms as transforms


TRANSFORM = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                                ])


class ProtestEvalDir(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.transform = TRANSFORM
        self.img_list = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        imgpath = os.path.join(self.img_dir, self.img_list[idx])
        image = pil_loader(imgpath)
        # we need this variable to check if the image is protest or not)
        sample = {"imgpath": imgpath, "image": image}
        sample["image"] = self.transform(sample["image"])
        return sample


class ProtestEvalImage(Dataset):
    def __init__(self, img_path):
        self.img_path = img_path

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image = pil_loader(self.img_path)
        # we need this variable to check if the image is protest or not)
        sample = {"imgpath": self.img_path, "image": image}
        sample["image"] = TRANSFORM(sample["image"])
        return sample


class FinalLayer(nn.Module):
    def __init__(self):
        super(FinalLayer, self).__init__()
        self.fc = nn.Linear(2048, 12)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)
        return out


def modified_resnet50():
    model = models.resnet50(pretrained=True)
    model.fc = FinalLayer()
    return model


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

