import carla
import random
import time
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from preprocess import extract_boxes, DataGenerator, calculate_iou, calculate_map

def spawn_vehicle(world, blueprint_library, transform):
    bp = blueprint_library.filter('model3')[0]  # Tesla Model 3 for example
    vehicle = world.spawn_actor(bp, transform)
    return vehicle

def get_camera_sensor(world, blueprint_library, vehicle):
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    return camera

def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    return array

def draw_bounding_box(image, box):
    start_point = (int(box[0]), int(box[1]))
    end_point = (int(box[2]), int(box[3]))
    color = (0, 255, 0)
    thickness = 2
    cv2.rectangle(image, start_point, end_point, color, thickness)

if __name__ == "__main__":
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = spawn_vehicle(world, blueprint_library, spawn_point)
    
    camera = get_camera_sensor(world, blueprint_library, vehicle)
    camera.listen(lambda image: process_image(image))
    
    model = load_model('object_detection_model.h5')

    while True:
        world.tick()
        image = camera.get()
        frame = process_image(image)
        input_image = cv2.resize(frame, (128, 128)) / 255.0
        input_image = np.expand_dims(input_image, axis=0)
        
        prediction = model.predict(input_image)[0]
        draw_bounding_box(frame, prediction)
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vehicle.destroy()
    camera.stop()
    camera.destroy()
    cv2.destroyAllWindows()
