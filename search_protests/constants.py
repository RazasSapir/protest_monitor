import os
from collections import namedtuple

# Paths
BASE_DIR = r""
BASE_DOWNLOAD_DIR = os.path.join(BASE_DIR, r"data\downloaded_images")
FINAL_PROTEST_DIR = os.path.join(BASE_DIR, r"data\protests_researched")
PLACES_PATH = os.path.join(FINAL_PROTEST_DIR, r"researched_locations.txt")
# ML weights paths
PROTEST_MODEL_PATH = os.path.join(BASE_DIR, r"protest_classification\weights\protest_weights.pth.tar")
YOLO_MODEL_PATH = os.path.join(BASE_DIR, r"counting_people\Yolo_detection\data\yolov3.weights")
CROWD_MODEL_PATH = os.path.join(BASE_DIR, r"counting_people\crowd_detection\weights\crownd_weights.pth.tar")

SLEEP_TIME = 2  # Seconds
PROTEST_THRESHOLD = 70  # Percentage
CYCLE_SIZE = 100  # Number of images downloaded in a cycle.
QUIET_DOWNLOAD = True

# Post namedtuple
POST_SCHEME = ["path", "shortcode", "datetime", "owner"]
Post = namedtuple("Post", POST_SCHEME)

# Protest namedtuple
PROTEST_SCHEME = ["post", "protest", "violence", "sign", "photo", "fire", "police", "children", "over_twenty",
                  "over_hundred", "flag", "night", "shouting"]
Protest = namedtuple("Protest", PROTEST_SCHEME)