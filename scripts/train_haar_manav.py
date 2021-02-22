from lns.common.preprocess import Preprocessor
from lns.haar.train import HaarTrainer
from lns.haar.eval import evaluate
from lns.haar.settings import HaarSettings

model_name = input("Enter model name: ")
dataset_y4signs = Preprocessor.preprocess('Y4Signs_1036_584_train_manav', force=True)
print(f"Classes before merge: {dataset_y4signs.classes}")

class_to_classify = ['nrt_nlt_sym', 'nrt_nlt_text', 'Stop', 'Yield'][HaarSettings.class_index]

dataset_y4signs = dataset_y4signs.merge_classes({
  "nrt_nlt_text": ['No_Right_Turn_Text', 'No_Left_Turn_Text'],
  "nrt_nlt_sym": ['No_Right_Turn_Sym', 'No_Left_Turn_Sym']
})
# dataset_y4signs.classes[0] = 'nrt_nlt_sym' # 4000 num_pos
# dataset_y4signs.classes[1] = 'nrt_nlt_text' # 4000 num_pos
# dataset_y4signs.classes[2] = 'Stop' # 2000
# dataset_y4signs.classes[3] = 'Yield' #2000
HaarSettings.class_index = dataset_y4signs.classes.index(class_to_classify)
index = dataset_y4signs.classes.index(class_to_classify)

print(f"Classes after merge: {dataset_y4signs.classes}")
print('Training model for: ' + dataset_y4signs.classes[HaarSettings.class_index])

trainer = HaarTrainer(name=model_name,
                        class_index = index,
                        dataset=dataset_y4signs, 
                        load=False) # Training from scratch



trainer.setup()
trainer.train()

print("Evaluating model for: " + str(dataset_y4signs.classes[HaarSettings.class_index]))

results = evaluate(data_path='/home/od/.lns-training/resources/processed/haar/Y4Signs_1036_584_test/annotations/{0}_positive'.format(class_to_classify),
                   model_path='/home/od/.lns-training/resources/trainers/haar/{0}/cascade/cascade.xml'.format(model_name),
                   num_neighbors=HaarSettings.min_neighbours)

file = open('/home/od/.lns-training/resources/trainers/haar/{}/results.txt'.format(model_name), "w")

tp, fp, precision, recall, f1_score = results
file.write("tp: {0}\nfp: {1}\nprecision: {2}\nrecall: {3}\nf1_score: {4}".format(tp, fp, precision, recall, f1_score))
file.close()
print('Training Completed Successfully. You can find trainer and results at: \n' + '/home/od/.lns-training/resources/trainers/haar/'+ model_name)

# from lns.haar.eval import evaluate
# results = evaluate(data_path='/home/od/.lns-training/resources/processed/haar/Y4Signs_1036_584_test/annotations/Stop_positive',
#                    model_path='/home/od/.lns-training/resources/trainers/haar/manav_test_run_30/cascade/cascade.xml')

# tp, fp, precision, recall, f1_score = results
# print("tp: {0}\nfp: {1}\nprecision: {2}\nrecall: {3}\nf1_score: {4}".format(tp, fp, precision, recall, f1_score))
