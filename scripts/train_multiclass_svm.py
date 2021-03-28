import os
from lns.haar.svm_classifier import SVMClassifier
from lns.haar.svm_preprocess_one_v_all import SVMProcessor
from lns.haar.svm_train import SVMTrainer
from lns.common.preprocess import Preprocessor

# TODO: update these constants after finished with data processing.
MODEL_OUTPUT_PATH = '/home/od/.lns-training/resources/trainers/haar'
PATH_TRAIN = '/home/od/.lns-training/resources/processed/haar/y4Signs_train_text_svm'
PATH_TEST = '/home/od/.lns-training/resources/processed/haar/y4Signs_test_text_svm'
MODEL_PATH = 'some_model_name'

def get_classes_to_classify(y4_signs_folder='Y4Signs_filtered_1036_584_train_split'):
    """
    Function to merge datasets and get classes indices to train
    """
    dataset_y4signs = Preprocessor.preprocess(y4_signs_folder)
    dataset_scalesigns = Preprocessor.preprocess('ScaleSigns')
    dataset = dataset_y4signs + dataset_scalesigns

    NRT_TEXT_IDX, NLT_TEXT_IDX = dataset.classes.index('No_Right_Turn_Text'), dataset.classes.index('No_Left_Turn_Text')
    RTO_TEXT_IDX, LTO_TEXT_IDX = dataset.classes.index('Right Turn Only (words)'), dataset.classes.index('Left Turn Only (words)')

    classify_nrt = (NRT_TEXT_IDX, (NLT_TEXT_IDX, RTO_TEXT_IDX, LTO_TEXT_IDX))
    classify_nlt = (NLT_TEXT_IDX, (NRT_TEXT_IDX, RTO_TEXT_IDX, LTO_TEXT_IDX))
    classify_rto = (RTO_TEXT_IDX, (NLT_TEXT_IDX, NRT_TEXT_IDX, LTO_TEXT_IDX))
    classify_lto = (LTO_TEXT_IDX, (NLT_TEXT_IDX, RTO_TEXT_IDX, NRT_TEXT_IDX))
    to_classify = [classify_nrt, classify_nlt, classify_rto, classify_lto]

    return dataset, to_classify


def train_one_vs_all(train_data, labels, model_name, 
                    processed_data_path=PATH_TRAIN, 
                    model_path=MODEL_OUTPUT_PATH):
    """
    Train a one vs all SVM model with data already partitioned properly located 
    in {processed_data_path}/{train_data} or /{labels}. 
    Output model will be saved under {model_path}/{model_name}/svm.xml
    """
    print(f"Training {model_name}")
    trainer = SVMTrainer(os.path.join(processed_data_path, train_data),
                         os.path.join(processed_data_path, labels),
                         os.path.join(model_path, model_name))
    trainer.setup()
    trainer.train()
    print("Done")


def evaluate_svm_models(test_data, labels, model_name, processed_data_path=PATH_TEST, model_path=MODEL_PATH):
    """
    Evaluate a one vs all SVM model with test data already partitioned properly located 
    in {processed_data_path}/{test_data} or /{labels}. Results will be printed out. 
    Load trained model from {model_path}/{model_name}/svm.xml
    """
    test = SVMClassifier(os.path.join(model_path, model_name))
    print(f"Evaluating {model_name} SVM")
    test.eval(os.path.join(processed_data_path, test_data),
              os.path.join(processed_data_path, labels))


PROCESS_TRAIN, PROCESS_TEST = True, False
TRAIN, TEST = False, False

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
    train_one_vs_all("", "", 'heddy_text_nrt_vs_all')



# dataset_1 = dataset_all.merge_classes({
#   "nrt_rest": ['No_Left_Turn_Text', 'Right Turn Only (words)', 'Left Turn Only (words)'],
# })
# print(dataset_1.classes.index('No_Right_Turn_Text'))
# print(dataset_1.annotations)


# -------- Class index constants --------
# ['No_Right_Turn_Text', 'No_Right_Turn_Sym', 'No_Left_Turn_Text', 'No_Left_Turn_Sym', 'Yield', 'Stop']
# ['25 mph', 'Regular Parking', 'Stop', 'Handicap Parking', 'arrow', '15 mph', 'Do Not Enter', '20 mph', 'words', '5 mph', '10 mph', 'Railroad Light Pair On', 'Railroad Sign', 'Railroad Light Pair Off', 'Speed Limit 20', 'Speed Limit 15', 'Speed Limit 10', 'Speed Limit 5', 'Right Turn Only (words)', 'Right Turn Only (arrow)', 'Left Turn Only (words)', 'Left Turn Only (arrow)']
