from lns.haar.svm_preprocess import SVMProcessor
from lns.haar.svm_train import SVMTrainer
from lns.common.preprocess import Preprocessor

processed_path = '/home/od/.lns-training/resources/processed/haar/y4Signs_train_svm'

dataset = Preprocessor.preprocess('Y4Signs_1036_584_train', force=True)
dataset.classes


to_classify = [(0, 2), (1, 3)] # class indices to differentiate between
#(0, 2) means that we are classifying between no_Right_Turn_Sym and no_Left_Turn_Sym

for a, b in to_classify:
    print("Found classes to differentiate between: {0} and {1}".format(a, b))

processor = SVMProcessor(processed_path, dataset, to_classify)
processor.preprocess()
processor.save_np_arrays()
data_path = '/home/od/.lns-training/resources/processed/haar/y4Signs_svm/No_Right_Turn_Sym_No_Left_Turn_Sym/data.npy'
labels = '/home/od/.lns-training/resources/processed/haar/y4Signs_svm/No_Right_Turn_Sym_No_Left_Turn_Sym/labels.npy'

processed_path_test = '/home/od/.lns-training/resources/processed/haar/y4Signs_test_svm'

# test_dataset = Preprocessor.preprocess('Y4Signs_1036_584_test', force=True)

# test_processor = SVMProcessor(processed_path_test, test_dataset, to_classify)
# test_processor.preprocess()
# test_processor.save_np_arrays()
