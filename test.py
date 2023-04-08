import pafy
import cv2
from acapture import acapture

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
    capture = acapture.open("test_video.mp4")
    while True:
        check, frame = capture.read()
        if check:
            frame = resize_frame(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)