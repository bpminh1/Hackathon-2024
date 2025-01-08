import cv2
import numpy as np

def find_placement(image_path, gripper_path):
    """
    Finds the optimal placement for a robot gripper on an object such that it can pick it up.

    Parameters:
    - image_path (str): Path to the image containing the object.
    - gripper_path (str): Path to the image containing the gripper with red dots.

    Returns:
    - tuple: (x-coordinate, y-coordinate, rotation angle) for placing the gripper.
    """
    # Process the image to extract contours (boundaries of the object)
    image, external_contours, internal_contours = process_image(image_path)

    # Calculate distances from the red dots on the gripper to its centroid
    distances = find_distances_from_red_dots_to_centroid(gripper_path)

    # Calculate the centroid of the tile (object) and the maximum search radius
    tile_centroid = calculate_tile_center(external_contours)
    max_radius = max(np.linalg.norm(np.array(point[0]) - tile_centroid) for point in external_contours)
    max_radius = int(np.ceil(max_radius))

    # Initialize the gripper centroid to the tile centroid for initial placement check
    gripper_centroid = tile_centroid
    ROTATION_ANGLE = 5  # Angle increment for rotation search
    RADIUS_CHANGE = 5   # Increment for radius search in expanding circles

    # Check if the gripper can be placed at the center of the object without red dots falling into holes
    for angle in range(0, 360, ROTATION_ANGLE):
        if check_red_dots_in_tile(distances, gripper_centroid, angle, external_contours, internal_contours):
            # Return placement with a small offset for better precision
            return float(gripper_centroid[0] - 10), float(gripper_centroid[1] - 10), angle

    # If center placement fails, check in expanding circles from the centroid
    for radius in range(1, max_radius, RADIUS_CHANGE):
        # Generate points on a circle with the given radius around the tile centroid
        circle = get_circle_coordinates(image, tile_centroid, radius)
        for point in circle:
            gripper_centroid = point  # Move gripper centroid to the current circle point

            # Verify the point is still within the tile boundary
            if is_point_inside_tile(gripper_centroid, external_contours, internal_contours):
                # Try all rotations for this position
                for angle in range(0, 360, ROTATION_ANGLE):    
                    if check_red_dots_in_tile(distances, gripper_centroid, angle, external_contours, internal_contours):
                        # Return placement if valid
                        return float(gripper_centroid[0] - 10), float(gripper_centroid[1] - 10), angle

    # Return default (0, 0, 0) if no valid placement is found
    return 0, 0, 0

def process_image(image_path):
    """
    Processes an image by loading it, adding padding, and extracting external and internal contours.

    Parameters:
    - image_path (str): Path to the image file to be processed.

    Returns:
    - tuple: (padded_image, external_contours, internal_contours)
    """
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply padding of 10 pixels on each side with a black border (value=0)
    padded_image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)

    # Find external contours of the padded image
    external_contours = external_contour(padded_image)

    # Find internal contours using the padded image and external contours
    internal_contours = internal_contour(padded_image, external_contours)

    # Return the processed image and the identified contours
    return padded_image, external_contours, internal_contours

def external_contour(image):
    """
    Extracts external contours from a given image after preprocessing.

    Parameters:
    - image (numpy.ndarray): Grayscale image to extract contours from.

    Returns:
    - list: A list of external contours found in the image.
    """
    # Preprocess the image to prepare for contour detection
    binary_external = preprocess_external(image)

    # Find external contours using OpenCV's findContours method
    external_contours, _ = cv2.findContours(
        binary_external, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    return external_contours

def preprocess_external(image):
    """
    Preprocesses an image for external contour detection by converting it to a binary image.

    Parameters:
    - image (numpy.ndarray): Grayscale image to preprocess.

    Returns:
    - numpy.ndarray: Binary image ready for contour detection.
    """
    # Apply Otsu's thresholding to convert the image to binary
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def internal_contour(image, external_contours):
    """
    Detects internal contours in an image by filtering out external contours.

    Parameters:
    - padded_image (numpy.ndarray): The padded grayscale image.
    - external_contours (list): List of external contours.

    Returns:
    - list: A list of detected internal contours.
    """
    # Preprocess the image for internal contour detection
    edges = preprocess_internal(image)

    # Detect contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create masks for filtering contours
    contour_mask = np.zeros_like(edges)
    cv2.drawContours(contour_mask, contours, -1, 255, thickness=cv2.FILLED)

    external_mask = np.zeros_like(edges)
    cv2.drawContours(external_mask, external_contours, -1, 255, thickness=5)

    # Subtract external edges from the contour mask
    result_mask = cv2.bitwise_and(contour_mask, cv2.bitwise_not(external_mask))

    # Find the final internal contours
    internal_contours, _ = cv2.findContours(result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return internal_contours

def preprocess_internal(image):
    """
    Preprocesses an image for internal contour detection using blurring and edge detection.

    Parameters:
    - padded_image (numpy.ndarray): The padded grayscale image.

    Returns:
    - numpy.ndarray: The processed binary edge image.
    """
    # Apply Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8, 8))
    equalized = clahe.apply(blurred)

    # Apply Canny edge detection
    edges = cv2.Canny(equalized, 50, 150)

    # Perform morphological closing to close gaps in edges
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return edges

def is_point_inside_tile(point, external_contours, internal_contours):
    """
    Check if a point is inside the tile but not inside any holes.

    Parameters:
    - point (tuple): The point to check.
    - external_contours (list): List of external contours.
    - internal_contours (list): List of internal contours.

    Returns:
    - bool: True if the point is inside the tile and not in any holes, False otherwise.
    """
    point = tuple(point)
    
    # Check if the point is inside any external contour (tile boundary)
    inside_tile = any(cv2.pointPolygonTest(contour, point, False) > 0 for contour in external_contours)
    
    # Check if the point is inside any internal contour (hole)
    inside_hole = any(cv2.pointPolygonTest(contour, point, False) >= 0 for contour in internal_contours)
    
    # Point is valid only if inside the tile and not inside any holes
    return inside_tile and not inside_hole

def check_red_dots_in_tile(distances, centroid, angle, external_contours, internal_contours):
    """
    Check if all red dots (gripper contact points) are validly placed within the tile.

    Parameters:
    - distances (numpy.ndarray): Array of distances from centroid to red dots.
    - centroid (numpy.ndarray): The centroid position.
    - angle (float): The angle to rotate red dots.
    - external_contours (list): List of external contours.
    - internal_contours (list): List of internal contours.

    Returns:
    - bool: True if all red dots are inside the tile and not in any holes, False otherwise.
    """
    # Rotate the red dot positions based on the specified angle
    distances = rotate_distances(distances, angle)

    # Calculate the absolute positions of the red dots relative to the centroid
    red_pixel_positions = centroid + distances

    # Verify if all red dots are within valid tile space
    for dot in red_pixel_positions:
        if not is_point_inside_tile(dot, external_contours, internal_contours):
            return False
    return True

def calculate_tile_center(external_contours):
    """
    Calculate the center of the largest contour from a list of external contours.
    The center is computed as the centroid of the contour.

    Parameters:
    - external_contours (list of np.ndarray): A list of external contours, where each contour is a numpy array of points.

    Returns:
    - tuple: The (x, y) coordinates of the centroid of the largest contour.
    """
    # Find the largest contour (assuming the tile is the largest object)
    largest_contour = max(external_contours, key=cv2.contourArea)

    # Calculate the centroid using image moments
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        # If the contour area is non-zero, calculate the centroid
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        # If the area is zero (the contour has no area), fallback to using the average of contour points
        cx, cy = np.mean(largest_contour[:, 0, :], axis=0).astype(int)

    return (cx, cy)


def get_circle_coordinates(image, center, radius):
    """
    Generate pixel coordinates forming a circle of one pixel width on a 2D numpy matrix.
    The circle is approximated by checking points within a bounding box around the circle's center.
    
    Parameters:
    - image (np.ndarray): A two-dimensional numpy matrix representing the image.
    - center (tuple): The (x, y) coordinates of the circle's center.
    - radius (int): The radius of the circle in pixels.

    Returns:
    - np.ndarray: An array of coordinates (x, y) forming the circle on the image.
    """
    # Get the height and width of the image
    height, width = image.shape
    circle_coordinates = []

    # Loop through a square region around the circle to avoid unnecessary checks
    for y in range(center[1] - radius, center[1] + radius + 1):
        for x in range(center[0] - radius, center[0] + radius + 1):
            # Check if the point is within the bounds of the image
            if 0 <= x < width and 0 <= y < height:
                # Check if the point lies on the circumference of the circle using the Euclidean distance formula
                distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                # If the distance is approximately equal to the radius, consider it part of the circle
                if abs(distance - radius) <= 0.5:  # Allow small error for pixel rounding
                    circle_coordinates.append((x, y))

    return circle_coordinates

def find_distances_from_red_dots_to_centroid(gripper_path):
    """
    Find the positions of red dots in the gripper image and calculate their distances from the centroid.
    
    Parameters:
    - gripper_path (str): The file path to the gripper image.

    Returns:
    - np.ndarray: An array of distances (x, y) from the centroid of the image to each red dot.
    """
    # Load the image from the specified path
    image = cv2.imread(gripper_path)

    # Find all positions where the red channel is dominant
    red_pixels = np.where((image[:, :, 2] > 100) &  # Red channel is high
                          (image[:, :, 2] > image[:, :, 1]) &  # Red > Green
                          (image[:, :, 2] > image[:, :, 0]))  # Red > Blue

    # Combine row and column indices into (x, y) positions of the red pixels
    red_pixel_positions = np.column_stack((red_pixels[1], red_pixels[0]))

    # Define the centroid of the gripper as the center of the image
    image_height, image_width = image.shape[:2]
    centroid_x = image_width / 2
    centroid_y = image_height / 2
    centroid = (centroid_x, centroid_y)

    # Calculate distances from the centroid to each red dot
    # The distances are represented as (x, y) offsets from the centroid
    distances = red_pixel_positions - centroid

    return distances


def rotate_distances(distances, angle):
    """
    Rotate the distances of red dots from the centroid by a specified angle.
    
    Parameters:
    - distances (np.ndarray): An array of distances (x, y) from the centroid to each red dot.
    - angle (float): The angle by which to rotate the distances, in degrees.

    Returns:
    - np.ndarray: An array of rotated distances (x, y).
    """
    # Convert angle from degrees to radians
    angle_rad = np.deg2rad(angle)

    # Define the rotation matrix for counterclockwise rotation
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])

    # Apply the rotation to each distance
    # The distances are transformed by multiplying with the rotation matrix
    return distances @ rotation_matrix.T