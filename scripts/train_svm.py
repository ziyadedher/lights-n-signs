from lns.haar.svm_preprocess import SVMProcessor
from lns.haar.svm_train import SVMTrainer
from lns.common.preprocess import Preprocessor

processed_path = '/home/od/.lns-training/resources/processed/haar/y4Signs_svm'

dataset = Preprocessor.preprocess('Y4Signs_1036_584_train', force=True)
dataset.classes
to_classfiy = [(0, 2), (1, 3)] 
for a, b in to_classfiy:
    print("Found classes to differentiate between: {0} and {1}".format(a, b))

processor = SVMProcessor(processed_path, dataset, to_classfiy)
processor.preprocess()
processor.save_np_arrays()

