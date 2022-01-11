import logging
import os.path

import cv2 as cv
import numpy as np
import time
import mediapipe as mp

from .plotting import plot
from .processing import smooth, get_video_dimensions

pose_labels = ["nose",
               "left eye inner",
               "left eye",
               "left eye outer",
               "right eye inner",
               "right eye",
               "right eye outer",
               "left ear",
               "right ear",
               "mouth left",
               "mouth right",
               "left shoulder",
               "right shoulder",
               "left elbow",
               "right elbow",
               "left wrist",
               "right wrist",
               "left pinky",
               "right pinky",
               "left index",
               "right index",
               "left thumb",
               "right thumb",
               "left hip",
               "right hip",
               "left knee",
               "right knee",
               "left ankle",
               "right ankle",
               "left heel",
               "right heel",
               "left foot index",
               "right foot index"]

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='./temp/tracking.log',
                    filemode='w')
logger = logging.getLogger('ball_tracking')
logger.setLevel(logging.DEBUG)


class Landmarks:
    def __init__(self, landmarks, frames: int):
        self.landmarks = landmarks
        self.frames = frames


class Results:
    def __init__(self, landmarks_class: Landmarks, bowling_speed: float):
        self.landmarks_class = landmarks_class
        self.bowling_speed = bowling_speed


class LandmarkAnalysis:
    def __init__(self, landmarks_class: Landmarks) -> np.ndarray:
        self.landmarks_class = landmarks_class

    def position_data(self, index: int = 0, visibility_threshold: float = 0.0, show_plot: bool = False):
        h, w = get_video_dimensions(4)
        frames = np.array(range(0, self.landmarks_class.frames)) / 240
        position = np.zeros((self.landmarks_class.frames, 4))

        for frame, landmark_frame in enumerate(self.landmarks_class.landmarks):
            if landmark_frame[index][3] < visibility_threshold:
                position[frame, :] = np.zeros((1, 4))
            else:
                position[frame, :] = np.multiply(landmark_frame[index], [w, h, 1, 1])

        if show_plot:
            plot(frames,
                 position,
                 "time (s)",
                 "position",
                 pose_labels[index].capitalize() + " position vs time",
                 ["x", "y", "z", "visibility"])

        return position

    def velocity_data(self, index: int = 0, visibility_threshold: float = 0.0, show_plot: bool = False):
        frames = np.array(range(0, self.landmarks_class.frames)) / 240 * 2
        position = self.position_data(index=index, visibility_threshold=visibility_threshold, show_plot=show_plot)
        velocity = smooth(np.gradient(position, frames, axis=0), 20)[:, 0:1] * 4

        if show_plot:
            plot(frames,
                 velocity,
                 "time (s)",
                 "velocity",
                 pose_labels[index].capitalize() + " velocity vs time",
                 ["x", "y"])

        return velocity


class PoseAnalysis:
    def __init__(self, video: bool = False):
        self.video = video
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.6)

    def get_landmarks(self, filepath: str) -> Landmarks:
        if os.path.exists(filepath[:-3] + "npy"):
            landmarks = np.load(filepath[:-3] + "npy")
            return Landmarks(landmarks=landmarks, frames=landmarks.shape[0])

        frame_count = 0
        cap = cv.VideoCapture(filepath)
        landmarks = []

        if not cap.isOpened():
            logger.error("cant open stream")
            return None

        while cap.isOpened():
            start = time.perf_counter()
            ret, frame = cap.read()

            if ret:
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                result = self.pose.process(frame_rgb)
                if result.pose_landmarks:
                    self.mp_draw.draw_landmarks(frame_rgb, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                    frame_landmarks = []
                    for landmark in result.pose_landmarks.landmark:
                        frame_landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

                    landmarks.append(frame_landmarks)

                if self.video:
                    cv.imshow('Frame', frame_rgb)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

            frame_count += 1

        cap.release()
        cv.destroyAllWindows()

        np.save(filepath[:-3] + "npy", landmarks)

        return Landmarks(landmarks=landmarks, frames=frame_count)
