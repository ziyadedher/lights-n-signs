from lns_common.preprocess.preprocess import Preprocessor

import cv2

def visualize(dataset: str) -> None:
    winname = "Images"
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname, 1200,1200)
    cv2.moveWindow(winname, 40,30)

    data = Preprocessor.preprocess(dataset)

    for img, boxes in data.annotations.items():
        print("Working on image: {}")

        image = cv2.imread(img)

        for box in boxes:
            label = data.classes[box['class']]
            cv2.rectangle(image, (box['x_min'], box['y_min']), \
                (box['x_max'], box['y_max']), (255,0,0), 10)
            cv2.putText(image, label, (box['x_min']-10,box['y_min']-10),\
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            print(box['x_min'], box['y_min'], box['x_max'], box['y_max'])

        cv2.imshow(winname, image)
        cv2.resizeWindow(winname, 1200,1200)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize('mturk')
