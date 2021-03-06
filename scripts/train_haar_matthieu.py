from lns.common.preprocess import Preprocessor
from lns.haar.train import HaarTrainer
from lns.haar.eval import evaluate
from lns.haar.settings import HaarSettings

# In settings
# class_index 0 = 'nrt_nlt_sym' # 4000 num_pos
# class_index 1 = 'nrt_nlt_rto_lto_text' # 7000 num_pos
# class_index 2 = 'Stop' # 2000 num_pos
# class_index 3 = 'Yield' # 2000 num_pos
class_to_classify = ['nrt_nlt_sym', 'nrt_nlt_rto_lto_text', 'Stop', 'Yield'][HaarSettings.class_index]


model_name = input("Enter model name: ")

# Get and merge data
dataset_y4signs = Preprocessor.preprocess('Y4Signs_1036_584_train_matthieu')
dataset_scalesigns = Preprocessor.preprocess('ScaleSigns')
print(f"Y4Signs classes before merge: {dataset_y4signs.classes}")
print(f"ScaleSigns classes before merge: {dataset_scalesigns.classes}")


dataset_all = dataset_y4signs + dataset_scalesigns
dataset_all = dataset_all.merge_classes({
  "nrt_nlt_rto_lto_text": ['No_Right_Turn_Text', 'No_Left_Turn_Text', 'Right Turn Only (words)', 'Left Turn Only (words)'],
  "nrt_nlt_sym": ['No_Right_Turn_Sym', 'No_Left_Turn_Sym']
})
print(f"Combined dataset classes after merge: {dataset_all.classes}")


HaarSettings.class_index = dataset_all.classes.index(class_to_classify)
print('Training model for: ' + dataset_all.classes[HaarSettings.class_index])

# Remove useless images to make preprocessing faster
print(f"The dataset has {len(dataset_all)} images in total")
irrelevant_imgs = []
for img, labels in dataset_all.annotations.items():
    labels = list(filter(
        lambda label: label.class_index == HaarSettings.class_index, labels)
        )
    if not labels:
        irrelevant_imgs.append(img)
    else:
        dataset_all.annotations[img] = labels

for img in irrelevant_imgs:
    del dataset_all.annotations[img]
    dataset_all.images.remove(img)
print(f"The dataset contains {len(irrelevant_imgs)} irrelevant images")
print(f"We'll keep the {len(dataset_all)} images belonging to {class_to_classify}")


# Train
trainer = HaarTrainer(name=model_name,
                        class_index=HaarSettings.class_index,
                        dataset=dataset_all, 
                        load=False) # Training from scratch
trainer.setup()
trainer.train()

# Test
print("Evaluating model for: " + str(dataset_all.classes[HaarSettings.class_index]))

results = evaluate(data_path=f'/home/od/.lns-training/resources/processed/haar/Y4Signs_1036_584_test/annotations/{class_to_classify}_positive',
                   model_path=f'/home/od/.lns-training/resources/trainers/haar/{model_name}/cascade/cascade.xml',
                   trainer_path=f'/home/od/.lns-training/resources/trainers/haar/{model_name}',
                   num_neighbors=HaarSettings.min_neighbours,
                   scale=HaarSettings.scale_factor
                   )
tp, fp, precision, recall, f1 = results

file = open(f'/home/od/.lns-training/resources/trainers/haar/{model_name}/results.txt', "w")
file.write(f"TP: {tp}\nFP: {fp}\nPrecision: {precision}\nRecall: {recall}\nF1 score: {f1}")
file.close()

print('Training Completed Successfully. You can find trainer and results at: \n' + '/home/od/.lns-training/resources/trainers/haar/'+ model_name)
