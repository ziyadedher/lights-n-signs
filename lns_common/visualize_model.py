from lns_common.preprocess.preprocess import Preprocessor

import cv2
from model.yolo import YOLO as model

def visualize(dataset: str) -> None:
    winname = "Images"
    cv2.namedWindow(winname)
    cv2.moveWindow(winname, 40,30)

    data = Preprocessor.preprocess(dataset)
    m = model()

    for img, boxes in data.annotations.items():
        print("Working on image: {}              \r".format(img), end="")

        image = cv2.imread(img)
        pred = m.predict(image)

        for box in pred:
            label = box.predicted_classes[0]
            x1 = box.bounding_box.left
            x2 = box.bounding_box.width + box.bounding_box.left
            y1 = box.bounding_box.top
            y2 = box.bounding_box.height + box.bounding_box.top
            cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.putText(image, label, (x1-10,y1-10),\
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        cv2.imshow(winname, image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize('mturk')
