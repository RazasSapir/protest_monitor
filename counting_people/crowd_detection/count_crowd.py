from matplotlib import pyplot as plt
from counting_people.crowd_detection.utils import CSRNet, Image
import torch
import numpy as np

from torchvision import transforms

TRANSFORM = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])


def get_model(weights_path):
    model = CSRNet(load_weights=True)
    model = model.cpu()
    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    return model


def count_crowd(model, img_path):
    img = TRANSFORM(Image.open(img_path).convert('RGB'))
    output = model(img.unsqueeze(0))
    amount = int(output.detach().cpu().sum().numpy())
    return output, amount


def plot_crowd(model, img_path):
    output, amount = count_crowd(model, img_path)
    print("Predicted Count : " + str(amount))
    temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2], output.detach().cpu().shape[3]))
    fig, a = plt.subplots(1, 2)
    a[1].set_title("Predicted Count : " + str(amount))
    a[1].imshow(temp)
    a[1].axis('off')
    a[0].imshow(plt.imread(img_path))
    a[0].axis('off')
    plt.show()


def main():
    pass


if __name__ == '__main__':
    main()






