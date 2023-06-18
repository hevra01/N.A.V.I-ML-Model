# YOLOv5 prediction => xmin        ymin        xmax        ymax  confidence  class    name

# This dictionary will be utilized while finding the distance of objects from the user.
# We are using the width of an object instead of its height because while capturing the
# frame, the complete height of the object might not be captured. The width is more likely
# to be appearing in full length.
objects_actual_width = {'person': 38.5, 'tv': 5, 'bicycle': 175, 'couch': 152, 'bus': 1200, 'car': 448, 'chair': 46,
                        'motorcycle': 221,
                        'traffic light': 40, 'bed': 141, 'bench': 114, 'dining table': 160, 'dog': 84, 'cat': 38}


# this function will calculate the distance of the object from the user
# by using the values of the focal_length, which will be given from the frontend,
# the actual_width of the object, which is already stored (the width of all the object of the same
# class are considered to be equal), and the perceived_width, which is returned from the ML model
def distance_calculation(focal_length, actual_width, perceived_width):
    distance = focal_length * (actual_width / perceived_width)
    return distance


# this function will take the prediction list from the ML model, which contains all the
# objects detected in a given frame. It also gets the actual_width of all the objects, which
# are already stored and the class_name dictionary, which specifies the name of the class
# given its index. E.g. class '0' represents a car.
def list_creation_objects_with_their_distances(predictions, objects_actual_width):
    # this list will be forwarded to the method that finds the positions of objects, where the position
    # information of the object will be appended to each dictionary element of the list.
    objects_with_positions = [[], [], []]
    instances = predictions[0].values.tolist()
    # this for loop is used to find the distances of the objects that are found by the ml model
    for instance in instances:
        actual_width = objects_actual_width[instance[6]]  # instance[6] represents the name
        perceived_width = instance[2] - instance[0]  # xmin is [0], xmax is [2]
        focal_length = 2  # take this value from the frontend
        distance = distance_calculation(focal_length, actual_width, perceived_width)
        distance = round(distance, 1)
        objects_with_positions[0].append(instance[6].capitalize())
        objects_with_positions[1].append(distance)

    return objects_with_positions


# calculating the position of the detected object https://www.techscience.com/cmc/v71n1/45390/html
# The whole image is divided into three positions such as top, center and bottom
# as row-wise and left, center and right as column-wise.

# this function will take the prediction list from the ML model, which contains all the
# objects detected in a given frame. additionally, it will also take the dictionary returned by the
# list_creation_objects_with_their_distances and append the position information to the object name and object distance.
# the return value will be a list of dictionaries, each dictionary holding the object name, object distance, and object position
def list_creation_objects_with_their_positions(predictions, objects_with_positions, FRAME_WIDTH, FRAME_HEIGHT):
    instances = predictions[0].values.tolist()

    # this for loop is used to find the distances of the objects that are found by the ml model
    for idx, instance in enumerate(instances):
        # the perceived width and height of the object
        width = instance[2] - instance[0]  # xmin is [0], xmax is [2]
        height = instance[3] - instance[1]  # ymin is [1], ymax is [3]

        # the x_center and y_center are the coordinates of the middle point of the bounding box
        # with respect to the origin which is (0,0).
        # x_center can be found by taking the average of
        # xmin which is the coordinate of the top left point of the bounding box with respect to the origin
        # and the top right point. the top right is equal to top left plus the width. same logic for y_center.
        x_center = ((2 * instance[0]) + (width)) / 2
        y_center = ((2 * instance[1]) + (height)) / 2

        if (x_center <= (FRAME_WIDTH / 3)) and (y_center <= FRAME_HEIGHT / 3):
            position = "Top left"
        elif (x_center <= (FRAME_WIDTH / 3)) and (y_center <= (FRAME_HEIGHT / 3) * 2):
            position = "Center left"
        elif (x_center <= (FRAME_WIDTH / 3)) and (y_center <= FRAME_HEIGHT):
            position = "Bottom left"
        elif (x_center <= (FRAME_WIDTH / 3) * 2) and (y_center <= FRAME_HEIGHT / 3):
            position = "Top center"
        elif (x_center <= (FRAME_WIDTH / 3) * 2) and (y_center <= (FRAME_HEIGHT / 3) * 2):
            position = "Center center"
        elif (x_center <= (FRAME_WIDTH / 3) * 2) and (y_center <= FRAME_HEIGHT):
            position = "Bottom center"
        elif (y_center <= FRAME_HEIGHT / 3):
            position = "Top right"
        elif (y_center <= (FRAME_HEIGHT / 3) * 2):
            position = "Center right"
        else:
            position = "Bottom right"

        objects_with_positions[2].append(position)

    return objects_with_positions


# this function first performs distance calculation and then performs position calculation (top left, buttom right, etc).
# its return value is a list of objects with their names, distances from the users, and position with respect to the user.
# e.g. {'object': car, 'distance': 2.5 meters, 'position': top left}
def calculate_position(predictions, objects_actual_width, FRAME_WIDTH, FRAME_HEIGHT):
    objects_with_positions = list_creation_objects_with_their_distances(predictions, objects_actual_width)
    objects_with_positions = list_creation_objects_with_their_positions(predictions, objects_with_positions,
                                                                        FRAME_WIDTH, FRAME_HEIGHT)
    return objects_with_positions
