from counting_people.Yolo_detection.utils.models import *
from counting_people.Yolo_detection.utils.utils import *
from counting_people.Yolo_detection.utils.datasets import *

from PIL import Image
import os
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

BASE_DIR = r""
MODEL_DEF = os.path.join(BASE_DIR, r"counting_people\Yolo_detection\data\yolov3.cfg")
CLASS_PATH = os.path.join(BASE_DIR, r"counting_people\Yolo_detection\data\coco.names")
CONFIDENCE_THRESH = 0.8
IOU_THRESHOLD = 0.4
NUM_WORKERS = 4
DIR_BATCH_SIZE = 16
IMG_SIZE = (416, 416)
IMG_WIDTH = 416

CLASSES = load_classes(CLASS_PATH)  # Extracts class labels from file
TENSOR = torch.FloatTensor

TRANSFORM = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()])


class EvalSingleImage(Dataset):
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


def get_model(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set up model
    model = Darknet(MODEL_DEF, img_size=IMG_SIZE).to(device)
    # Load darknet weights
    model.load_darknet_weights(weights_path)
    model.eval()  # Set in evaluation mode
    return model


def count_people(model, img_path):
    dataset = EvalSingleImage(img_path)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=NUM_WORKERS)
    for sample in data_loader:
        img_path, input_data = sample['imgpath'], sample['image']
        input_var = Variable(input_data.type(TENSOR))
        with torch.no_grad():
            detections = model(input_var)
            detections = non_max_suppression(detections, CONFIDENCE_THRESH, IOU_THRESHOLD)
    sum_people = 0
    if detections[0] is not None:
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0]:
            if CLASSES[int(cls_pred)] == "person":
                sum_people += 1
    return sum_people


def run_yolo_dir(model, dir_path):
    dataloader = DataLoader(
        ImageFolder(dir_path, img_size=IMG_SIZE),
        batch_size=DIR_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(TENSOR))
        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, CONFIDENCE_THRESH, IOU_THRESHOLD)
        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)
    return imgs, img_detections


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def plot_detection(imgs, img_detections):
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        print("(%d) Image: '%s'" % (img_i, path))
        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, IMG_WIDTH, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (CLASSES[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(x1, y1, s=CLASSES[int(cls_pred)], color="white", verticalalignment="top", bbox={"color": color, "pad": 0})

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f"{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()


def main():
    pass


if __name__ == '__main__':
    main()