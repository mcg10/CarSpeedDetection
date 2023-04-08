import pafy
import cv2
from ffpyplayer.player import MediaPlayer
import numpy as np

url = "https://www.youtube.com/watch?v=5_XSYlAfJZM"  # Tilton


def resize_frame(frame):
    scale_percent = 60
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized


if __name__ == '__main__':
    url = "https://www.youtube.com/watch?v=5_XSYlAfJZM"  # Tilton
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")
    player = MediaPlayer("test_video.mp4")
    while True:
        frame, _ = player.get_frame()
        if frame:
            img, t = frame
            w = img.get_size()[0]
            h = img.get_size()[1]
            arr = np.uint8(np.asarray(list(img.to_bytearray()[0])).reshape(h, w, 3))
            cv2.imshow('frame', arr)
            cv2.waitKey(1)