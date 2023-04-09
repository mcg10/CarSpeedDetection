import pafy
import cv2
from ffpyplayer.player import MediaPlayer
import numpy as np
import imageio.v3 as iio

url = "https://www.youtube.com/watch?v=5_XSYlAfJZM"  # Tilton


def resize_frame(frame):
    scale_percent = 60
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized


if __name__ == '__main__':
    for frame in iio.imiter("test_video.mp4", plugin="pyav"):
        cv2.imshow('frame', frame)
        cv2.waitKey(1)