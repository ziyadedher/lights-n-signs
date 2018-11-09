"""Video Stream Tester.

This script is for using a video video stream from your webcam
as the input to a model.
"""
import cv2

from common.model import Model

CLAHE = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))


def test_video_stream(model: Model) -> None:
    """This function tests a model instance on your webcam's video stream."""
    cap = cv2.VideoCapture(0)

    while 1:
        _, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = CLAHE.apply(gray)
        try:
            detected_list = model.predict(gray)
            for obj in detected_list:
                label = obj.predicted_classes[0]
                b = obj.bounding_box
                cv2.rectangle(
                    img,
                    (b.left, b.top),
                    (b.left + b.width, b.top + b.height),
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    img,
                    label,
                    (b.left, b.height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
            cv2.imshow('img', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        except Exception:
            print("Error encountered in model's predict method")
            cap.release()
            cv2.destroyAllWindows()
            raise

    cap.release()
    cv2.destroyAllWindows()