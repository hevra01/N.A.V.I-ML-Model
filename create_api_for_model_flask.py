from flask import Flask, request, jsonify, g as app_ctx
import pickle
import io
import time
from position_calculator import calculate_position
import PIL

# change classes based on kitti
KITTI_CLASSES = [
    "car",
    "cyclist",
    "pedestrian",
    "van",
    "truck",
    "tram",
    "person sitting"
]

objects_actual_width = {'person': 38.5, 'car': 448,}

navi_app = Flask(__name__)

# loads the ml model from a pickle file
with open('dist_yolo_model.pk', 'rb') as file:
    model = pickle.load(file)
    file.close()


# flask's before_request and after_request decorator to measure time taken for a request to complete.
# this is to test API response time
@navi_app.before_request
def logging_before():
    # Store the start time for the request
    app_ctx.start_time = time.perf_counter()


@navi_app.after_request
def logging_after(response):
    # Get total time in milliseconds
    total_time = time.perf_counter() - app_ctx.start_time
    time_in_ms = int(total_time * 1000)
    # Log the time taken for the endpoint 
    print('Response time => ', time_in_ms, 'ms')
    return response


@navi_app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return "Please send post request"

    elif request.method == "POST":
        frame = request.files.get('frame')  # get the frame sent by the API request
        im_bytes = frame.read()  # convert the file into byte stream
        image = PIL.Image.open(io.BytesIO(im_bytes))  # convert the byte stream into
        image.show()
        image_width, image_height = image.size

        prediction = model(image)
        objects_with_positions = calculate_position(prediction.pandas().xyxy, objects_actual_width, image_width,
                                                    image_height)

        data = {
            "objects_with_positions": objects_with_positions,
        }

        return jsonify(data)


navi_app.run(port=5000, host='0.0.0.0', debug=False)
