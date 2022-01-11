import numpy as np
import json


def smooth(points: np.ndarray, box_points: int = 10):
    smoothed = np.ones(np.shape(points))
    for i in range(0, np.shape(points)[1]):
        box = np.ones(box_points) / box_points
        smoothed[:, i] = np.convolve(points[:, i], box, mode='same')
    return smoothed


def get_video_dimensions(camera_distance):
    with open("./information/config.json", "r+") as read:
        data = json.load(read)
        dhr = data["distance_height_ratio"]
        ar = data["aspect_ratio"]
        return camera_distance * dhr, camera_distance * dhr / ar
