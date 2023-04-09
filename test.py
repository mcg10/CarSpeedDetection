import cv2
from imutils.video import FPS
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
    fps = FPS().start()
    count = 0
    for frame in iio.imiter("tilton_detection.avi", plugin="pyav"):
        if count % 3 != 0:
            count += 1
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        fps.update()
        count += 1
    fps.stop()
    print(fps.fps())
