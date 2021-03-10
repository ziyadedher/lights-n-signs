import os
from lns.common.preprocess import Preprocessor
from lns.haar.train import HaarTrainer
from lns.haar.eval import evaluate
from lns.haar.settings import HaarSettings
from lns.haar.process import HaarProcessor



# Must be False if you don't want to disrupt any other training.
# Can only be True if no other training instances are running
FORCE_PREPROCESSING = False


class_to_classify = ['nrt_nlt_sym', 'nrt_nlt_text', 'Stop', 'Yield'][HaarSettings.class_index]
model_name = input("Enter model name: ")

dataset_y4signs = Preprocessor.preprocess('Y4Signs_filtered_1036_584_train_split', force=True) # force = True will give stats
print(f"Classes before merge: {dataset_y4signs.classes}")

# merge classes
dataset_y4signs = dataset_y4signs.merge_classes({
  "nrt_nlt_text": ['No_Right_Turn_Text', 'No_Left_Turn_Text'],
  "nrt_nlt_sym": ['No_Right_Turn_Sym', 'No_Left_Turn_Sym']
})
# dataset_y4signs.classes[0] = 'nrt_nlt_sym' # 2000 num_pos
# dataset_y4signs.classes[1] = 'nrt_nlt_text' # 2000 num_pos
# dataset_y4signs.classes[2] = 'Stop' # 2000
# dataset_y4signs.classes[3] = 'Yield' #2000

new_class_index = dataset_y4signs.classes.index(class_to_classify)
HaarSettings.class_index = new_class_index

print(f"Classes after merge: {dataset_y4signs.classes}")

# training routine
print('Training model for: ' + dataset_y4signs.classes[new_class_index])

trainer = HaarTrainer(name=model_name,
                        class_index = new_class_index,
                        dataset=dataset_y4signs, 
                        load=False, # Training from scratch
                        forcePreprocessing = FORCE_PREPROCESSING) # True means preprocessing from scratch

trainer.setup()
trainer.train()


# evaluation routine
print("Evaluating model for: " + str(dataset_y4signs.classes[HaarSettings.class_index]))

# IMPORTANT: must preprocess the test folder before running this script. This name should exist in '/home/od/.lns-training/resources/processed/haar/'
eval_preprocessed_folder = 'Y4Signs_filtered_1036_584_test_split'

processed_path_for_eval = '/home/od/.lns-training/resources/processed/haar/{0}'.format(eval_preprocessed_folder)

if not os.path.exists(eval_preprocessed_folder):
  print("did not find preprocessed test dataset, preprocessing from scratch")
  # TODO: preprocess the evaluation dataset
  dataset_y4signs_eval = Preprocessor.preprocess('Y4Signs_filtered_1036_584_test_split', force=True) # force = True will give stats
  dataset_y4signs_eval = dataset_y4signs_eval.merge_classes({
  "nrt_nlt_text": ['No_Right_Turn_Text', 'No_Left_Turn_Text'],
  "nrt_nlt_sym": ['No_Right_Turn_Sym', 'No_Left_Turn_Sym']
  })
  _ = HaarProcessor.process(dataset_y4signs_eval, force=True)

  print("Preprocessing complete! Evaluating...")
  





# evaluate model after training and save visualisations and numeric data
results = evaluate(data_path='/home/od/.lns-training/resources/processed/haar/{0}/annotations/{1}_positive'.format(eval_preprocessed_folder, class_to_classify),
                   model_path='/home/od/.lns-training/resources/trainers/haar/{0}/cascade/cascade.xml'.format(model_name),
                   trainer_path= '/home/od/.lns-training/resources/trainers/haar/{0}'.format(model_name),
                   num_neighbors=HaarSettings.min_neighbours,
                   scale=HaarSettings.scale_factor)


print('Training Completed Successfully. You can find trainer and results at: \n' + '/home/od/.lns-training/resources/trainers/haar/'+ model_name)
