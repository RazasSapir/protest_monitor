import cv2
import imutils
from sign_detection.utils.visualize import imshow
from sign_detection.constants import DEBUG, MIN_SQUARE_AREA_RATIO, MAX_SQUARE_AREA_RATIO
from sign_detection.intersection_lines.find_quadrilaterals import only_parallelograms

APPROX_EPSILON_RATIO = 0.1


def find_contour(thresh):
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    print("Found " + str(len(contours)) + " contours.") if DEBUG else None

    large_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        image_area = thresh.shape[0] * thresh.shape[1]
        if MIN_SQUARE_AREA_RATIO * image_area < area < MAX_SQUARE_AREA_RATIO * image_area:
            large_contours.append(c)
    if DEBUG:
        copy = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(copy, large_contours, -1, (0, 0, 255), 4)
        imshow(copy, "contour")
        print(str(len(large_contours)) + " of them are large enough.")
    return large_contours


def get_squares(contours):
    squares = []
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, APPROX_EPSILON_RATIO * perimeter, True)
        if len(approx) == 4:
            squares.append(approx.reshape(4, 2))
    print("Found " + str(len(squares)) + " squares.") if DEBUG else None
    parallel = only_parallelograms(squares)
    print(str(len(parallel)) + " of them are parallelograms.") if DEBUG else None
    return parallel
