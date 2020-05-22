import os
from counting_people.crowd_detection import count_crowd
from counting_people.Yolo_detection import YOLO_detection

BASE_DIR = r""
SSD_MODEL_PATH = os.path.join(BASE_DIR, r"counting_people\SSD_counter\data\yolov3.weights")
CROWD_MODEL_PATH = os.path.join(BASE_DIR, r"\counting_people\crowd_detection\weights\crownd_weights.pth.tar")


def count_people(img_path, ssd_model, crowd_model):
    sum_ssd = YOLO_detection.count_people(ssd_model, img_path)
    if sum_ssd == 0:
        _, amount = count_crowd.count_crowd(crowd_model, img_path)
        return amount
    return sum_ssd


def get_models(ssd_weights, crowd_weights):
    return YOLO_detection.get_model(ssd_weights), count_crowd.get_model(crowd_weights)


def main():
    pass


if __name__ == '__main__':
    main()