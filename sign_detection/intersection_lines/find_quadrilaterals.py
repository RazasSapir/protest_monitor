import numpy as np
from bentley_ottmann.planar import segments_intersections
import itertools
import math
from sign_detection.utils import visualize
from sign_detection.constants import DEBUG

# Resize Lines / Segments
LINE_TO_SEGMENT_SIZE = 1000
EXTEND_SEGMENTS_WITH = 500

MIN_SIDE_LEN = 15

# Unique Segments
MAX_ANGLE_BETWEEN_SAME_SEGMENTS = 10 * math.pi / 180  # Radians
MAX_DIST_BETWEEN_SAME_SEGMENTS = 20  # Pixels

# Collinear
MAX_ANGLE_BETWEEN_COLLINEAR_VECTORS = 5 * math.pi / 180  # Radians

# Parallel Lines
MAX_ANGLE_BETWEEN_PARALLEL_VECTORS = 20 * math.pi / 180  # Radians

# Unique Lines
MIN_DIFF_ANGLE = 10 * math.pi / 180  # Radians
MIN_DIFF_RADIUS = 15

image_to_draw = None


def get_segments(lines):
    """
    list of lines -> list of segments
    :param lines: list of lines in a form of list<(rho, theta)>
    :return: list of segments of size 2 * SEGMENT_SIZE of form list<((x1, y1), (x2, y2))>
    """
    segments = []
    for line in lines:
        if len(line[0]) == 4:
            x1, y1, x2, y2, = line[0]
        else:
            rho, theta = line[0]
            horizontal_coeff = np.cos(theta)
            vertical_coeff = np.sin(theta)
            x0, y0 = horizontal_coeff * rho, vertical_coeff * rho
            x1 = int(x0 + LINE_TO_SEGMENT_SIZE * (-vertical_coeff))
            y1 = int(y0 + LINE_TO_SEGMENT_SIZE * (horizontal_coeff))
            x2 = int(x0 - LINE_TO_SEGMENT_SIZE * (-vertical_coeff))
            y2 = int(y0 - LINE_TO_SEGMENT_SIZE * (horizontal_coeff))
        segments.append(((x1, y1), (x2, y2)))
    return segments


def get_intersections(segments):
    """
    list of segments -> dictionary of {intersection point:segment indexes}
    :param segments: list of segments of form of form list<((x1, y1), (x2, y2))>
    :return: dictionary of points to the segments that cross it:
    dictionary<(point<Fraction, Fraction>:list<int>)>
    """
    return segments_intersections(segments)


def four_pt_list(points):
    """
    list of points -> list of all possible combinations of 4 points (no repetition)
    :param points: list of point<Fraction, Fraction>
    :return: iterator of all combination of four points<Fraction, Fraction>
    """
    return itertools.combinations(points, 4)


def clockwiseangle_and_distance(point, origin=None, refvec=(0, 1)):
    """
    point -> vector, used as a sorting key for the list of points by angle
    sorted(points, key=clockwiseangle_and_distance)
    :param point: point<Fraction, Fractoin>
    :param origin: the origin of the vector: default is the origin
    :param refvec: vector the reference the angle
    :return: angle(radians), vector length
    """
    # Vector between point and the origin: v = p - o
    vector = [point[0] - origin[0], point[1] - origin[1]]
    lenvector = math.hypot(vector[0], vector[1])  # Length of vector: ||v||
    if lenvector == 0:  # If length is zero there is no angle
        return -math.pi, 0
    normalized = [vector[0] / lenvector, vector[1] / lenvector]  # Normalize vector: v/||v||
    dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]  # x1*x2 + y1*y2
    diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]  # x1*y2 - y1*x2
    angle = math.atan2(diffprod, dotprod)
    # Negative angles: need to subtract them from 2*pi (360 degrees)
    if angle < 0:
        return 2 * math.pi + angle, lenvector
    # angle is the primary sorting criterium then length
    return angle, lenvector


def only_quads(point_lists, intersection_dict):
    """
    List of four points -> list of quadrilaterals on lines
    - Eliminate lines, triangle and quadrilaterals with the wrong lines
    :param point_lists: list of tuples of 4 Point<Fraction, Fraction>
    :param intersection_dict: dictionary of intersection point and the lines the cross them
    { Point<Fraction, Fraction>: list<int> }
    :return:
    """
    quadrilaterals = []
    first_it, second_it = itertools.tee(point_lists)
    len_fours = sum(1 for q in first_it)
    print("There are " + str(len_fours) + " groups of four points.") if DEBUG else None
    for k, points in enumerate(second_it):
        origin = points[0]
        points = sorted(points, key=lambda point: clockwiseangle_and_distance(point, origin))
        visualize.show_points(image_to_draw, points)
        is_quadrilateral = True
        for i in range(len(points)):  # On the Lines
            if i == 0:
                if set(next(iter(intersection_dict[points[0]]))).isdisjoint(
                        next(iter(intersection_dict[points[len(points) - 1]]))):
                    is_quadrilateral = False
            else:
                if set(next(iter(intersection_dict[points[i - 1]]))).isdisjoint(
                        next(iter(intersection_dict[points[i]]))):
                    is_quadrilateral = False
                    break
        for i in range(len(points)):  # Large enough sides
            if i == 0:
                if math.hypot(points[0][0] - points[len(points) - 1][0],
                              points[0][1] - points[len(points) - 1][1]) < MIN_SIDE_LEN:  # Finish circle
                    is_quadrilateral = False
            else:
                if math.hypot(points[i - 1][0] - points[i][0], points[i - 1][1] - points[i][1]) < MIN_SIDE_LEN:
                    is_quadrilateral = False
                    break
        for three_list in itertools.combinations(points, 3):  # Not Collinear
            if parallel_lines(three_list[0], three_list[1], three_list[0], three_list[2], MAX_ANGLE_BETWEEN_COLLINEAR_VECTORS):
                is_quadrilateral = False
                break
        if is_quadrilateral:
            quadrilaterals.append(points)
    return quadrilaterals


def fractions_to_int(fraction_tuple_list):
    """
    list of tuple of Point<Fraction, Fraction> -> np.array of Point<int, int>
    :param fraction_tuple_list:
    :return:
    """
    new_tuple_list = []
    for quad in fraction_tuple_list:
        new_quad = []
        for point in quad:
            new_quad.append([int(point[0]), int(point[1])])
        new_tuple_list.append(new_quad)
    return np.array(new_tuple_list)


def parallel_lines(p00, p01, p10, p11, angle):
    """
    Line 1: p01 - p00
    Line 2: p11 - p10
    Returns True is Line 1 and Line 2 are parrallel else False
    :param p00:
    :param p01:
    :param p10:
    :param p11:
    :return:
    """
    vect1 = np.array([float(p01[0]) - float(p00[0]), float(p01[1]) - float(p00[1])])
    vect2 = np.array([float(p11[0]) - float(p10[0]), float(p11[1]) - float(p10[1])])
    angle_between = angle_between_vectors(vect1, vect2)
    if angle_between < angle or angle_between > math.pi - angle:
        return True
    return False


def only_parallelograms(quads):
    """
    list of quadrilaterals -> list of parallelograms
    :param quads: each quadrilateral is a list of four Point<Fraction, Fraction>
    :return:
    """
    parallelograms = []
    for parall in quads:
        is_parallelograms = True
        p1, p2, p3, p4 = parall
        if not parallel_lines(p1, p2, p3, p4, MAX_ANGLE_BETWEEN_PARALLEL_VECTORS):
            is_parallelograms = False
        if not parallel_lines(p1, p4, p2, p3, MAX_ANGLE_BETWEEN_PARALLEL_VECTORS):
            is_parallelograms = False
        if is_parallelograms:
            parallelograms.append(parall)
    return parallelograms


def find_unique_lines(lines):
    """
    list of lines (represented by rho and theta) -> list of unique lines
    Eliminate lines with similar rho and theta
    :param lines:
    :return:
    """
    if lines is None or len(lines) == 0:
        return []
    unique_lines = []
    for l1 in lines:
        is_unique = True
        rho1, theta1 = l1[0]
        for l2 in unique_lines:
            rho2, theta2 = l2[0]
            if abs(theta1 - theta2) < MIN_DIFF_ANGLE and abs(rho1 - rho2) < MIN_DIFF_RADIUS:
                is_unique = False
                break
        if is_unique:
            unique_lines.append(l1)
    return unique_lines


def angle_between_vectors(vect1, vect2):
    unit_vector_1 = vect1 / np.linalg.norm(vect1)
    unit_vector_2 = vect2 / np.linalg.norm(vect2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(np.clip(dot_product, -1.0, 1.0))


def connect_same_segments(segments):
    if segments is None or len(segments) == 0:
        return []
    uniques_segmenst = [segments[0]]
    for s1 in segments[1:]:
        is_unique = True
        s1p1, s1p2 = s1
        for s2 in uniques_segmenst:
            s2p1, s2p2 = s2
            is_angle_close = parallel_lines(s1p1, s1p2, s2p1, s2p2, MAX_DIST_BETWEEN_SAME_SEGMENTS)
            if not is_angle_close:
                continue
            distance_line_point = np.linalg.norm(np.cross(s1p2 - s1p1, s1p1 - s2p1)) / np.linalg.norm(s1p2 - s1p1)
            is_distance_close = distance_line_point < MAX_DIST_BETWEEN_SAME_SEGMENTS
            if is_angle_close and is_distance_close:
                is_unique = False
                max_len = 0
                max_couple = [s1p1, s1p2]
                for couple_points in itertools.combinations([s1p1, s1p2, s2p1, s2p2], 2):
                    p1, p2 = couple_points
                    dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                    if dist > max_len:
                        max_len = dist
                        max_couple = couple_points
                temp_unique = []
                for u in uniques_segmenst:
                    if not np.array_equal(u, s2):
                        temp_unique.append(u)
                uniques_segmenst = temp_unique
                uniques_segmenst.append(np.asarray(max_couple))
                continue
        if is_unique:
            uniques_segmenst.append(s1)
    return uniques_segmenst


def extend_segments(segments):
    if segments is None or len(segments) == 0:
        return []
    extended = []
    for s in segments:
        p1, p2 = s
        vect = p2 - p1
        unit_vector_1 = vect / np.linalg.norm(vect)
        add_vect = unit_vector_1 * EXTEND_SEGMENTS_WITH
        p1 = p1 - add_vect
        p2 = p2 + add_vect
        extended.append(tuple([tuple([int(p1[0]), int(p1[1])]), tuple([int(p2[0]), int(p2[1])])]))
    return extended


def get_parallelograms_by_lines(lines):
    """
    list of lines (by rho and theta) -> np.array of parallelograms by four Point<int,int>
    :param lines:
    :return:
    """
    print_list = []
    if lines is None or len(lines) == 0:
        return []
    print("There are: " + str(len(lines)) + " lines.") if DEBUG else None
    unique_lines = find_unique_lines(lines)
    print(str(len(unique_lines)) + " are unique lines.") if DEBUG else None
    if len(unique_lines) == 0:
        return []
    segments = get_segments(unique_lines)
    print("Calculated Segments.") if DEBUG else None
    return __get_parallelograms__(segments)


def get_parallelograms_by_segments(segments, image=None):
    global image_to_draw
    image_to_draw = image
    if segments is None or len(segments) == 0:
        return []
    print("There are: " + str(len(segments)) + " segments.") if DEBUG else None
    unique_segments = connect_same_segments(segments)
    print(str(len(unique_segments)) + " are unique segments.") if DEBUG else None
    if len(unique_segments) == 0:
        return []
    extended_segments = extend_segments(unique_segments)
    return __get_parallelograms__(extended_segments)


def __get_parallelograms__(segments):
    intersection_dict = get_intersections(segments)
    print("Found " + str(len(intersection_dict)) + " intersection points.") if DEBUG else None
    four_lists = four_pt_list(intersection_dict)
    quadrilaterals = only_quads(four_lists, intersection_dict)
    print("Found " + str(len(quadrilaterals)) + " quadrilaterals.") if DEBUG else None
    if len(quadrilaterals) == 0:
        return []
    parallelograms = only_parallelograms(quadrilaterals)
    print(str(len(parallelograms)) + " are parallelograms.") if DEBUG else None
    return fractions_to_int(parallelograms)
