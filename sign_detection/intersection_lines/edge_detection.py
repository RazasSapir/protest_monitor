import cv2
import numpy as np
from sign_detection.intersection_lines import find_quadrilaterals
from sign_detection.utils import visualize
from sign_detection.constants import DEBUG


CANNY_WEAK_THRESH = 0
CANNY_STRONG_THRESH = 0
HOUGH_THRESH_BASE = 100

MIN_LINE_LEN = 20
MAX_LINE_GAP = 200


def canny_edges(image):
    edges = cv2.Canny(image, CANNY_WEAK_THRESH, CANNY_STRONG_THRESH)
    return edges


def detect_lines(edges):
    hough_thresh = HOUGH_THRESH_BASE
    lines = cv2.HoughLines(edges, 1, np.pi / 180, HOUGH_THRESH_BASE)  # shape = (None, 1, 2)
    len_lines = len(find_quadrilaterals.find_unique_lines(lines))
    while (lines is None or len_lines < 4) and hough_thresh > 50:
        lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_thresh - 2)
        len_lines = len(find_quadrilaterals.find_unique_lines(lines))
        hough_thresh -= 2
    if lines is None:
        return []
    while len_lines > 8:
        lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_thresh + 1)
        len_lines = len(find_quadrilaterals.find_unique_lines(lines))
        hough_thresh += 1
    print("Thresh value: " + str(hough_thresh)) if DEBUG else None
    return lines if lines is not None else []


def get_segments(edges, hough_thresh, min_line_len=MIN_LINE_LEN, max_line_gap=MAX_LINE_GAP):
    segments = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_thresh, minLineLength=min_line_len, maxLineGap=max_line_gap)
    if segments is None or len(segments) == 0:
        return []
    else:
        return segments.reshape(-1, 2, 2)


def detect_segments(edges):
    hough_thresh = HOUGH_THRESH_BASE
    segments = get_segments(edges, hough_thresh)
    len_segments = len(find_quadrilaterals.connect_same_segments(segments))
    while (segments is None or len_segments < 4) and hough_thresh > 50:
        segments = get_segments(edges, hough_thresh - 2)
        len_segments = len(find_quadrilaterals.connect_same_segments(segments))
        hough_thresh -= 2
    if segments is None:
        return []
    while len_segments > 12:
        segments = get_segments(edges, hough_thresh + 1)
        len_segments = len(find_quadrilaterals.connect_same_segments(segments))
        hough_thresh += 1
    print("Thresh value: " + str(hough_thresh)) if DEBUG else None
    return segments


def detect_squares_by_segments(image):
    edges = canny_edges(image)
    segments = detect_segments(edges)
    if DEBUG:
        copy_segmensts = find_quadrilaterals.extend_segments(find_quadrilaterals.connect_same_segments(segments))
        image_with_lines = visualize.show_segments(edges, copy_segmensts)  # Visualise for debugging
    quads = find_quadrilaterals.get_parallelograms_by_segments(segments)
    return quads


def detect_squares_by_lines(image):
    edges = canny_edges(image)
    lines = detect_lines(edges)
    if DEBUG:
        copy_lines = find_quadrilaterals.find_unique_lines(lines)
        visualize.show_lines(image, copy_lines)
    quads = find_quadrilaterals.get_parallelograms_by_lines(lines)
    return quads