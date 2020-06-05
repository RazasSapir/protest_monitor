import numpy as np
import cv2
from matplotlib.colors import hsv_to_rgb
from matplotlib import pyplot as plt
from sign_detection.constants import DEBUG

COLORS_BOUNDARIES = {
    'red': (((0, 100, 120), (15, 255, 255)),  ((165, 100, 100), (180, 255, 255))),  # Red has 2 variations
    'yellow': ((10, 100, 120), (35, 255, 255)),
    'green': ((30, 100, 120), (85, 255, 255)),
    'blue': ((80, 100, 120), (145, 255, 255)),
    'pink': ((140, 100, 170), (170, 255, 255)),
    'white': ((0, 0, 150), (180, 100, 255)),
    'black': ((0, 0, 0), (180, 255, 70))
}


def show_color(colors):
    fig, ax = plt.subplots(len(colors), 2, sharex='col', sharey='row')
    for i, c in enumerate(colors):
        color = colors[c]
        if c == 'red':
            color = color[0]
        lo_square = np.full((10, 10, 3), color[0], dtype=np.uint8) / 255.0
        do_square = np.full((10, 10, 3), color[1], dtype=np.uint8) / 255.0
        ax[i, 0].imshow(hsv_to_rgb(lo_square))
        ax[i, 0].set_title(c + ": low")
        ax[i, 1].imshow(hsv_to_rgb(do_square))
        ax[i, 1].set_title(c + ": high")
    plt.show()


def split_masks(image):
    masks_dict = {}
    hsv_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
    for color_key in COLORS_BOUNDARIES:
        if not color_key == 'red':
            low_color, high_color = COLORS_BOUNDARIES[color_key]
            mask = cv2.inRange(hsv_image, low_color, high_color)
        else:
            first_bound, second_bound = COLORS_BOUNDARIES[color_key]
            mask1 = cv2.inRange(hsv_image, first_bound[0], first_bound[1])
            mask2 = cv2.inRange(hsv_image, second_bound[0], second_bound[1])
            mask = mask1 + mask2
        masks_dict[color_key] = mask
    return masks_dict


def get_masks(image):
    masks_dict = split_masks(image)
    masks = {}
    for mask_key in masks_dict:
        masks[mask_key] = cv2.merge((masks_dict[mask_key], masks_dict[mask_key], masks_dict[mask_key]))
    return masks


def split_colors(image):
    channels_dict = {}
    masks_dict = split_masks(image)
    for mask_key in masks_dict:
        filtered = cv2.bitwise_and(image, image, mask=masks_dict[mask_key])
        channels_dict[mask_key] = filtered
    if DEBUG:
        for channel_key in channels_dict:
            cv2.imshow(channel_key, channels_dict[channel_key])
            cv2.waitKey(0)

    return channels_dict


def main():
    show_color(COLORS_BOUNDARIES)


if __name__ == "__main__":
    main()
