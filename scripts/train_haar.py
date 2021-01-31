from lns.common.preprocess import Preprocessor
from lns.haar.train import HaarTrainer


dataset_y4signs = Preprocessor.preprocess('Y4Signs')
print(f"Classes before merge: {dataset_y4signs.classes}")

dataset_y4signs = dataset_y4signs.merge_classes({
  "nrt_nlt_text": ['No_Right_Turn_Text', 'No_Left_Turn_Text'],
  "nrt_nlt_sym": ['No_Right_Turn_Sym', 'No_Left_Turn_Sym']
})
print(f"Classes after merge: {dataset_y4signs.classes}")

trainer = HaarTrainer(name="matthieu_haar_y4signs_1",
                        dataset=dataset_y4signs, 
                        load=False) # Training from scratch

trainer.setup()
trainer.train()
