from lns.common.preprocess import Preprocessor
from lns.haar.train import HaarTrainer


dataset_y4signs = Preprocessor.preprocess('Y4Signs')
print(f"Classes before merge: {dataset_y4signs.classes}")

dataset_y4signs = dataset_y4signs.merge_classes({
  "nrt_nlt_text": ['No_Right_Turn_Text', 'No_Left_Turn_Text'],
  "nrt_nlt_sym": ['No_Right_Turn_Sym', 'No_Left_Turn_Sym']
})
print(f"Classes after merge: {dataset_y4signs.classes}")

# Initializing the Haar trainer calls _process from lns/haar/process.py
# It saves images one by one, which takes 1h30min. This is not necessary if the trainer was previously initialized (and the images were already saved)
# Hence, a hack around this is to comment out lines 76-99 of that file
trainer = HaarTrainer(name="matthieu_haar_y4signs_1",
                        dataset=dataset_y4signs, 
                        load=False) # Training from scratch

trainer.setup()
trainer.train()
