import os
from lns.haar.svm_classifier import SVMClassifier
from lns.haar.svm_preprocess import SVMProcessor
from lns.haar.svm_train import SVMTrainer
from lns.common.preprocess import Preprocessor

model_path = '/home/od/.lns-training/resources/trainers/haar'
processed_path_train = '/home/od/.lns-training/resources/processed/haar/y4Signs_train_svm'
processed_path_test = '/home/od/.lns-training/resources/processed/haar/y4Signs_test_svm'
nlrt_signs_data = 'No_Right_Turn_Sym_No_Left_Turn_Sym/data.npy'
nlrt_signs_label = 'No_Right_Turn_Sym_No_Left_Turn_Sym/labels.npy'
nlrt_text_data = 'No_Right_Turn_Text_No_Left_Turn_Text/data.npy'
nlrt_text_label = 'No_Right_Turn_Text_No_Left_Turn_Text/labels.npy'
PROCESS_TRAIN, PROCESS_TEST = False, False
TRAIN_SYM, TRAIN_TEXT = True, False
TEST_SYM, TEST_TEXT = True, False

if PROCESS_TRAIN:
    dataset = Preprocessor.preprocess('Y4Signs_filtered_1036_584_train_split', force=True)

    to_classify = [(0, 2), (1, 3)]  # class indices to differentiate between
    # (0, 2) means that we are classifying between no_Right_Turn_Sym and no_Left_Turn_Sym

    for a, b in to_classify:
        print("Found classes to differentiate between: {0} and {1}".format(a, b))

    processor = SVMProcessor(processed_path_train, dataset, to_classify)
    processor.preprocess(force=True)
    processor.save_np_arrays()

if TRAIN_SYM:
    print("Training No_Right_Turn_Sym_No_Left_Turn_Sym")
    trainer = SVMTrainer(os.path.join(processed_path_train, nlrt_signs_data),
                         os.path.join(processed_path_train, nlrt_signs_label),
                         os.path.join(model_path, 'heddy_lrt_sign'))
    trainer.setup()
    trainer.train()
    print("Done")
if TRAIN_TEXT:
    print("Training No_Right_Turn_Text_No_Left_Turn_Text")
    trainer = SVMTrainer(os.path.join(processed_path_train, nlrt_text_data),
                         os.path.join(processed_path_train, nlrt_text_label),
                         os.path.join(model_path, 'heddy_lrt_text'))
    trainer.setup()
    trainer.train()
    print("Done")

if PROCESS_TEST:
    dataset = Preprocessor.preprocess('Y4Signs_filtered_1036_584_test_split', force=True)

    to_classify = [(0, 2), (1, 3)]  # class indices to differentiate between
    # (0, 2) means that we are classifying between no_Right_Turn_Sym and no_Left_Turn_Sym

    for a, b in to_classify:
        print("Found classes to differentiate between: {0} and {1}".format(a, b))

    processor = SVMProcessor(processed_path_test, dataset, to_classify)
    processor.preprocess(force=False)
    processor.save_np_arrays()

if TEST_SYM:
    test = SVMClassifier(os.path.join(model_path, 'heddy_lrt_sign'))
    print("Evaluating NoRight Turn Sym NoLeft Turn Sym")
    test.eval(os.path.join(processed_path_test, nlrt_signs_data),
              os.path.join(processed_path_test, nlrt_signs_label))
if TEST_TEXT:
    test = SVMClassifier(os.path.join(model_path, 'heddy_lrt_text'))
    print("Evaluating NoRight Turn Text NoLeft Turn Text")
    test.eval(os.path.join(processed_path_test, nlrt_text_data),
              os.path.join(processed_path_test, nlrt_text_label))


# test_dataset = Preprocessor.preprocess('Y4Signs_1036_584_test', force=True)

# test_processor = SVMProcessor(processed_path_test, test_dataset, to_classify)
# test_processor.preprocess()
# test_processor.save_np_arrays()
