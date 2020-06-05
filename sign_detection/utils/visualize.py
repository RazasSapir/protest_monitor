import cv2
from matplotlib import pyplot as plt
from sign_detection.intersection_lines import find_quadrilaterals
from sign_detection.utils import colors
from sign_detection.constants import DEBUG


def print_coordinates(event, x, y):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(str(x) + ", " + str(y))


def show_lines(image, lines):
    if lines is None or len(lines) == 0:
        return
    lines = find_quadrilaterals.find_unique_lines(lines)
    segments = find_quadrilaterals.get_segments(lines)
    return show_segments(image, segments)


def show_segments(image, segments):
    if segments is None or len(segments) == 0:
        return
    copy = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for s in segments:
        cv2.line(copy, tuple(s[0]), tuple(s[1]), (0, 255, 0), 2)
    """cv2.namedWindow('image')
    cv2.setMouseCallback('image', print_coordinates)"""
    imshow(copy)
    return copy


def show_points(image, points):
    if image is None:
        return
    if image.shape[-1] == 1:
        copy = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        copy = image.copy()
    for p in points:
        cv2.circle(copy, (int(p[0]), int(p[1])), 7, (0, 0, 255), -1)
    imshow(copy)
    plot_images([image, copy])
    return copy


def show_squares(image, squares):
    cv2.drawContours(image, squares, -1, (0, 0, 255), 4)
    return image


def imshow(image, title='image', force=False):
    if DEBUG or force:
        new_size = (500, int(500 / image.shape[1] * image.shape[0]))
        image = cv2.resize(image, new_size)
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def show_masks(image, color_masks, in_color=False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, axs = plt.subplots(2, 4)
    counter = 0
    colors_list = list(colors.COLORS_BOUNDARIES.keys())
    for i in range(2):
        for j in range(4):
            axs[i, j].set_title(colors_list[counter])
            if in_color:
                axs[i, j].imshow(cv2.bitwise_and(image, image, mask=color_masks[colors_list[counter]][:, :, 0]))
            else:
                axs[i, j].imshow(color_masks[colors_list[counter]])
            axs[i, j].axis('off')
            counter += 1
    plt.show()


def plot_images(images):
    fig, axs = plt.subplots(1, len(images))
    for i, img in enumerate(images):
        axs[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[i].axis('off')
    plt.show()
