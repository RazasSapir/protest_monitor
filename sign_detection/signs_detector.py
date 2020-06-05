import cv2
import os
from shapely.geometry.polygon import Polygon
from sign_detection.utils import straighten_signs
from sign_detection.OCR.OCR import get_text_from_image
from sign_detection.utils import colors, visualize
from sign_detection.utils.visualize import imshow
from sign_detection.intersection_lines import edge_detection
from sign_detection.contour import contour_detection
from sign_detection.constants import *

COLORS = colors.COLORS_BOUNDARIES.keys()


def preprocess_image(image):
    blur_kernel = int(np.ceil(image.shape[0] * BLUR_RATIO) + int(np.ceil(image.shape[1] * BLUR_RATIO))) // 4 * 2 + 1
    blurred = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    sharpen = cv2.filter2D(blurred, -1, SHARPEN_KERNEL)
    return sharpen


def process_mask(image, color='Image'):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imshow(gray, color)

    open_width = int(np.ceil(image.shape[0] * OPEN_RATIO) // 2 * 2 + 1)
    open_height = int(np.ceil(image.shape[1] * OPEN_RATIO) // 2 * 2 + 1)
    open_kernel = np.ones((open_width, open_height), np.uint8)
    img_open = cv2.morphologyEx(gray, cv2.MORPH_OPEN, open_kernel)
    imshow(img_open, color + ' opening')

    close_width = int(np.ceil(image.shape[0] * CLOSING_RATIO) // 2 * 2 + 1)
    close_height = int(np.ceil(image.shape[1] * CLOSING_RATIO) // 2 * 2 + 1)
    closing_kernel = np.ones((close_width, close_height), np.uint8)
    img_no_line_gaps = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, closing_kernel)
    imshow(img_no_line_gaps, color + " closing")

    return img_no_line_gaps


def analyze_signs_from_image(image, method):
    processed = preprocess_image(image)
    imshow(processed, "Processed")
    masks_dict = colors.get_masks(processed)
    squares = []
    for color in COLORS:
        current_channel = masks_dict[color]
        processed = process_mask(current_channel, color=color)
        if method == DETECT_BY_LINES:
            signs = edge_detection.detect_squares_by_lines(processed)
        elif method == DETECT_BY_SEGMENTS:
            signs = edge_detection.detect_squares_by_segments(processed)
        elif method == DETECT_BY_CONTOUR:
            contour = contour_detection.find_contour(processed)
            signs = contour_detection.get_squares(contour)
        else:
            print("Does not recognize method")
            raise AttributeError
        if not len(signs) == 0:
            squares.append(signs)
    if len(squares) == 0:
        return []
    return [item for sublist in squares for item in sublist]


def remove_fully_contained_signs(squares):
    signs_do_not_contain = []
    for first_sign in squares:
        contains = False
        s1 = Polygon(first_sign)
        for second_sign in squares:
            if (first_sign == second_sign).all():
                continue
            s2 = Polygon(second_sign)
            contains = s1.contains(s2)
            if contains:
                break
        if not contains:
            signs_do_not_contain.append(first_sign)
    return signs_do_not_contain


def compare_methods(image):
    signs_segments = analyze_signs_from_image(image, method=DETECT_BY_SEGMENTS)
    image_segments = visualize.show_squares(image.copy(), signs_segments)
    signs_lines = analyze_signs_from_image(image, method=DETECT_BY_LINES)
    image_lines = visualize.show_squares(image.copy(), signs_lines)
    signs_contour = analyze_signs_from_image(image, method=DETECT_BY_CONTOUR)
    image_contour = visualize.show_squares(image.copy(), signs_contour)
    visualize.plot_images([image_segments, image_lines, image_contour])


def analyze_dir(dir_path, out_path):
    for img_name in os.listdir(dir_path):
        if not img_name.split('.')[1] in ['jpg', 'png']:
            continue
        img_path = os.path.join(dir_path, img_name)
        print(img_path)
        image = cv2.imread(img_path)
        image = analyze_image(image)
        cv2.imwrite(os.path.join(out_path, img_name), image)


def analyze_image(image):
    imshow(image, "Original")
    signs_list = analyze_signs_from_image(image, method=DETECT_BY_SEGMENTS)
    image = visualize.show_squares(image, signs_list)
    imshow(image, "Final")
    return image


def find_signs(image_path):
    image = cv2.imread(image_path)
    signs = analyze_signs_from_image(image, method=DETECT_BY_CONTOUR) + analyze_signs_from_image(image, method=DETECT_BY_SEGMENTS)
    signs = remove_fully_contained_signs(signs)
    rectangle_signs = []
    for coordinates in signs:
        wrapped = straighten_signs.four_point_transform(image, coordinates)
        rectangle_signs.append(cv2.cvtColor(wrapped, cv2.COLOR_BGR2RGB))
    signs_with_text = []
    for sign in rectangle_signs:
        if not get_text_from_image(sign) == "":
            signs_with_text.append(sign)
            imshow(sign, "Sign")
    return signs_with_text


def main():
    pass


if __name__ == "__main__":
    main()
