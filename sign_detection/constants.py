import numpy as np

DEBUG = False


# Square Methods
DETECT_BY_LINES = 'lines'
DETECT_BY_SEGMENTS = 'segments'
DETECT_BY_CONTOUR = 'contour'

# Process Image
BLUR_RATIO = 0.03
SHARPEN_KERNEL = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

# Process Mask
OPEN_RATIO = 0.01
CLOSING_RATIO = 0.02

# Eliminate Squares
MIN_SQUARE_AREA_RATIO = 0.015  # of the Image Area
MAX_SQUARE_AREA_RATIO = 0.95  # of the Image Area
