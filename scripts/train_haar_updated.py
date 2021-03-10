from lns.common.preprocess import Preprocessor
from lns.haar.train import HaarTrainer
from lns.haar.eval import evaluate
from lns.haar.settings import HaarSettings



# Must be False if you don't want to disrupt any other training.
# Can only be True if no other training instances are running
FORCE_PREPROCESSING = False

# class_index 0 = 'nrt_nlt_sym' # 4000 num_pos
# class_index 1 = 'nrt_nlt_rto_lto_text' # 7000 num_pos
# class_index 2 = 'Stop' # 2000 num_pos
# class_index 3 = 'Yield' # 2000 num_pos
class_to_classify = ['nrt_nlt_sym', 'nrt_nlt_rto_lto_text', 'Stop', 'Yield'][HaarSettings.class_index]
model_name = input("Enter model name: ")

# Get and merge data
dataset_y4signs = Preprocessor.preprocess('Y4Signs_filtered_1036_584_train_split', force=True) # force = True will give stats
dataset_scalesigns = Preprocessor.preprocess('ScaleSigns')
print(f"\nY4Signs classes before merge: {dataset_y4signs.classes}")
print(f"ScaleSigns classes before merge: {dataset_scalesigns.classes}\n")

dataset_all = dataset_y4signs + dataset_scalesigns
dataset_all = dataset_all.merge_classes({
  "nrt_nlt_rto_lto_text": ['No_Right_Turn_Text', 'No_Left_Turn_Text', 'Right Turn Only (words)', 'Left Turn Only (words)'],
  "nrt_nlt_sym": ['No_Right_Turn_Sym', 'No_Left_Turn_Sym']
})
print(f"Combined dataset classes after merge: {dataset_all.classes}\n")

HaarSettings.class_index = dataset_all.classes.index(class_to_classify)
print('Training model for: ' + dataset_all.classes[HaarSettings.class_index])

trainer = HaarTrainer(name=model_name,
                        class_index=HaarSettings.class_index,
                        dataset=dataset_all, 
                        load=False, # Training from scratch
                        forcePreprocessing=FORCE_PREPROCESSING) 

trainer.setup()
trainer.train()


# evaluation routine
print("Evaluating model for: " + str(dataset_all.classes[HaarSettings.class_index]))

# IMPORTANT: must preprocess the test folder before running this script. This name should exist in '/home/od/.lns-training/resources/processed/haar/'
eval_preprocessed_folder = 'Y4Signs_filtered_1036_584_test_split'

# evaluate model after training and save visualisations and numeric data
results = evaluate(data_path=f'/home/od/.lns-training/resources/processed/haar/{eval_preprocessed_folder}/annotations/{class_to_classify}_positive',
                   model_path=f'/home/od/.lns-training/resources/trainers/haar/{model_name}/cascade/cascade.xml',
                   trainer_path=f'/home/od/.lns-training/resources/trainers/haar/{model_name}',
                   num_neighbors=HaarSettings.min_neighbours,
                   scale=HaarSettings.scale_factor)


print('Training Completed Successfully. You can find trainer and results at: \n' + f'/home/od/.lns-training/resources/trainers/haar/{model_name}')
