from scripts.player import initialise
from scripts.analysis import pose
import matplotlib.pyplot as plt


def video_analysis(player_id, video_id):
    player = player_id
    video = video_id
    file = f"./profiles/{player}/videos/{player}_{video}.mp4"

    pose_analysis = pose.PoseAnalysis(video=True)
    landmarks = pose_analysis.get_landmarks(filepath=file)

    if landmarks:
        landmarks_analysis = pose.LandmarkAnalysis(landmarks_class=landmarks)
        vel_wrist = landmarks_analysis.velocity_data(index=16, visibility_threshold=-100.0, show_plot=True)


def new_profile():
    initialise.player_menu()


if __name__ == '__main__':
    video_analysis(4, 1)
