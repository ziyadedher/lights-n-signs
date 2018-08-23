import glob
import random
import pickle
import darknet

if __name__ == "__main__":
    net = darknet.load_net("cfg/yolov3.cfg", "yolov3.weights", 0)
    meta = darknet.load_meta("cfg/coco.data")
    test_dir = "/Users/arkark/autoronto/lights-n-signs-training/common/data/dayTrain/**/frames/*"
    test_files = glob.glob(test_dir)

    random.seed(0)
    random.shuffle(test_files)

    with open("../common/preprocess/annotations.pkl", "rb") as f:
        annotations = pickle.load(f)

    false_negatives, false_positives, true_positives, true_negatives, false_alarms = 0, 0, 0, 0, 0
    for f in test_files:
        if not f in annotations: continue
        r = darknet.detect(net, meta, f, thresh=0.65)
        detected = []
        correct_number = sum(len(annotations[f][i]) for i in annotations[f])
        for detection in r:
            if detection[0] == 'traffic light':
                detected.append(detection)
        if correct_number == len(detected):
            if correct_number == 0:
                print("Correctly detected no lights")
                true_negatives += 1
            else:
                print("Correctly detected all lights!")
                true_positives += len(detected)
        elif correct_number > len(detected):
            print("Missed some, detected {} when there were {}".format(len(detected), correct_number))
            true_positives += len(detected)
            false_negatives += correct_number - len(detected)
        elif correct_number < len(detected):
            if correct_number == 0:
                print("False alarm, detected when there were none")
                false_alarms += 1
            print("Some extra ones, detected {} when there were {}".format(len(detected), correct_number))
            true_positives += correct_number
            false_positives += len(detected) - correct_number
        print(detected, annotations[f])
        print("True positives: {}".format(true_positives))
        print("True negatives: {}".format(true_negatives))
        print("False positives: {}".format(false_positives))
        print("False negatives: {}".format(false_negatives))
        print("False alarms: {}".format(false_alarms))

