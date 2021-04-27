import os
from lns.haar.svm_classifier import SVMClassifier, f1_score
from lns.haar.svm_preprocess_multiclass import SVMProcessor
from lns.haar.svm_train import SVMTrainer
from lns.common.preprocess import Preprocessor
import cv2 as cv
import numpy as np
from collections import Counter
import pickle

MODEL_OUTPUT_PATH = '/home/od/.lns-training/resources/trainers/haar'
PATH_TRAIN = '/home/od/.lns-training/resources/processed/haar/y4Signs_train_text_svm_48_singleModel'
PATH_TEST = '/home/od/.lns-training/resources/processed/haar/y4Signs_test_text_svm_48_singleModel'

def get_classes_to_classify(y4_signs_folder='Y4Signs_filtered_1036_584_train_split'):
    """
    Function to merge datasets and get classes indices to train
    """
    dataset_y4signs = Preprocessor.preprocess(y4_signs_folder)
    dataset_scalesigns = Preprocessor.preprocess('ScaleSigns')
    dataset = dataset_y4signs + dataset_scalesigns

    NRT_TEXT_IDX, NLT_TEXT_IDX = dataset.classes.index('No_Right_Turn_Text'), dataset.classes.index('No_Left_Turn_Text')
    RTO_TEXT_IDX, LTO_TEXT_IDX = dataset.classes.index('Right Turn Only (words)'), dataset.classes.index('Left Turn Only (words)')

    # classify_nrt = (NRT_TEXT_IDX, (NLT_TEXT_IDX, RTO_TEXT_IDX, LTO_TEXT_IDX))
    to_classify = [NRT_TEXT_IDX, NLT_TEXT_IDX, RTO_TEXT_IDX, LTO_TEXT_IDX]
    # classify_nlt = (NLT_TEXT_IDX, (NRT_TEXT_IDX, RTO_TEXT_IDX, LTO_TEXT_IDX))
    # classify_rto = (RTO_TEXT_IDX, (NLT_TEXT_IDX, NRT_TEXT_IDX, LTO_TEXT_IDX))
    # classify_lto = (LTO_TEXT_IDX, (NLT_TEXT_IDX, RTO_TEXT_IDX, NRT_TEXT_IDX))
    # to_classify = [classify_nrt, classify_nlt, classify_rto, classify_lto]

    return dataset, to_classify


def train_multiclass(train_data, labels, model_name, 
                    processed_data_path=PATH_TRAIN, 
                    model_path=MODEL_OUTPUT_PATH):
    """
    Train a one vs all SVM model with data already partitioned properly located 
    in {processed_data_path}/{train_data} or /{labels}. 
    Output model will be saved under {model_path}/{model_name}/svm.xml
    """
    print(f"\nTraining {model_name}")
    trainer = SVMTrainer(os.path.join(processed_data_path, train_data),
                         os.path.join(processed_data_path, labels),
                         os.path.join(model_path, model_name))
    trainer.setup()
    trainer.train()


def evaluate_svm_models(test_data, labels, model_name, processed_data_path=PATH_TEST, model_path=MODEL_OUTPUT_PATH):
    """
    Evaluate a one vs all SVM model with test data already partitioned properly located 
    in {processed_data_path}/{test_data} or /{labels}. Results will be printed out. 
    Load trained model from {model_path}/{model_name}/svm.xml
    """
    path = os.path.join(model_path, model_name)
    print(f"\nEvaluating {model_name} SVM at {path}")
    test = SVMClassifier(path, input_size=(48, 48))

    test_data_path = os.path.join(processed_data_path, test_data)
    test_label_path = os.path.join(processed_data_path, labels) # absolute paths
    test_mapping_path = os.path.join(processed_data_path, "labels_mapping.pkl") 

    label_mapping = pickle.load(open(test_mapping_path, 'rb')) # mapping of labels and their respective classes.

    val_data = np.load(test_data_path, allow_pickle=True) # load test data
    val_data = np.float32(val_data).reshape((-1, test.input_size[0] * test.input_size[1]))
    val_labels = np.load(test_label_path, allow_pickle=True) # load labels
    val_labels = np.int32(val_labels).reshape((val_labels.shape[0],1))

    # counters for metrics of individual class labels
    tps = Counter()
    fps = Counter()
    fns = Counter()

    predicted_results = test.predict(val_data)[1]

    tp, fp = 0, 0
    for i in range(val_data.shape[0]):
        if predicted_results[i] == val_labels[i]:
            tps[predicted_results[i]] += 1 # add to the true positives of the predicted label
        else:
            fps[predicted_results[i]] += 1 # add to the false positives of the predicted label
            fns[val_labels[i]] += 1 # add to the false negatives of the ground truth label

    for lab in label_mapping.keys()):
        print("label: " + str(label_mapping[lab]))
        precision = float(tps[lab]) / float(fps[lab] + tps[lab])
        recall = float(tps[lab]) / float(fns[lab] + tps[lab])
        print("TP: {}\FP: {}\Precision: {:.2f}\Recall: {:.2f}\nF1 score: {:.2f}".format(tps[lab], fps[lab], precision, recall, f1_score(precision, recall)))
    

def _data_path():
    # signs = ["No_Right_Turn_Text", "No_Left_Turn_Text", "Right Turn Only (words)", "Left Turn Only (words)"]
    # models = ["nrt", "nlt", "rto", "lto"]
    return [
        os.path.join("data.npy"),
        os.path.join("labels.npy"),
        "helen_text_multiclass"]


def _test_joint(input_size=(48, 48)):
    signs = ["No_Right_Turn_Text", "No_Left_Turn_Text", "Right Turn Only (words)", "Left Turn Only (words)"]
    models = ["nrt", "nlt", "rto", "lto"]

    tp, fp, total = 0, 0, 0

    # i = the current ground truth
    for i in range(4):
        val_path = os.path.join(PATH_TEST, signs[i], "data.npy")
        labels_path = os.path.join(PATH_TEST, signs[i], "labels.npy")
        val_data = np.load(val_path, allow_pickle=True)
        val_data = np.float32(val_data).reshape((-1, input_size[0] * input_size[1]))
        val_labels = np.load(labels_path, allow_pickle=True)
        val_labels = np.int32(val_labels).reshape((val_labels.shape[0],1))

        res = []
        for j in range(4):
            model_path = os.path.join(MODEL_OUTPUT_PATH, "helen_text_{}_vs_rest".format(models[j]))

            trained_model = cv.ml.SVM_load(model_path + "/svm.xml")
            predicted_results = trained_model.predict(val_data)[1]
            res.append(predicted_results)

        bad_count = 0  # the number of samples where the number of SVMs that classify "positive" != 1
        for v in range(val_data.shape[0]):  # loop through all val samples
            if val_labels[v] == 0:  # if the label is 0 which means is a ground truth positive sample
    
                results_to_verify = [res[0][v], res[1][v], res[2][v], res[3][v]]  # for this sample, the values outputted by each SVM
                # print(results_to_verify)  # debug note: res[0] mostly giving 1, res[1] mostly 0 but some 1, res[2] / res[3] are basically all 0 from what I see (3 is worst)
                if np.sum(results_to_verify) != 3:
                    bad_count += 1
                predicted = np.argmin(results_to_verify)  # 0 is positive so take minimum confidence
                if predicted == i:  # if predicted is the ground truth
                    # if i == 2:
                    #     cv.imwrite("test/label-" + str(i) + "-" + str(v) + "true" + str(predicted) + ".jpg", np.resize(val_data[v], (32, 32)))
                    tp += 1
                else:
                    fp += 1
                # total += 1
        
        print("BAD COUNT:", bad_count)  # around 10% of NT predictions are bad, 25% of TO predictions are bad


    precision = float(tp) / float(fp + tp)
    recall = float(tp) / float(fp + tp)

    print(total)
    print("TP: {}\FP: {}\Precision: {:.2f}\Recall: {:.2f}\nF1 score: {:.2f}".format(tp, fp, precision, recall, f1_score(precision, recall)))


PROCESS_TRAIN, PROCESS_TEST = False, True # DO NOT TURN ON, only if data re-processing is required
TRAIN, TEST, TEST_JOINT = False, True, False

if PROCESS_TRAIN:
    dataset, to_classify = get_classes_to_classify()
    processed_path_train = PATH_TRAIN
    processor = SVMProcessor(processed_path_train, dataset, to_classify)
    processor.preprocess(force=True)
    processor.save_np_arrays()

if PROCESS_TEST:
    dataset, to_classify = get_classes_to_classify(y4_signs_folder='Y4Signs_filtered_1036_584_test_split')
    processed_path_test = PATH_TEST
    processor = SVMProcessor(processed_path_test, dataset, to_classify)
    processor.preprocess(force=True)
    processor.save_np_arrays()

if TRAIN:
    # SVM Multi-Class
    files_path = _data_path()
    train_multiclass(files_path[0],files_path[1],files_path[2])

if TEST:
    files_path = _data_path()
    evaluate_svm_models(files_path[0],files_path[1],files_path[2])

if TEST_JOINT:
    _test_joint()

