import cv2
import os

from shutil import copyfile

"""
Initializing the Haar trainer calls _process from lns/haar/process.py
It saves images one by one, which takes 2min for lisa_signs but 1h30min for Y4Signs (lisa_signs has 6000 images size 500kB (dimensions 1024x522) but Y4Signs has 11000 images size 8MB (dimensions 5184x2920))

This script copies data from Y4Signs into Y4Signs_1036_584 while resizing all images to 1036 x 584.

It is VERY IMPORTANT to preserve the aspect ratio since the object coordinates are fractions of the image sizes.
"""

NEW_SIZE = (1036, 584)
ROOT = "/mnt/ssd6/Y4Signs"
ROOT_RESIZED = f"/mnt/ssd6/Y4Signs_{NEW_SIZE[0]}_{NEW_SIZE[1]}"


# Construct set01, ..., set62
for content1 in os.listdir(ROOT):
    path1 = os.path.join(ROOT, content1)
    path1_resized = os.path.join(ROOT_RESIZED, content1)
    print(path1_resized)

    os.makedirs(path1_resized, exist_ok=True)
    
    # Construct obj_train_data/, obj.data, obj.names, train.txt
    for content2 in os.listdir(path1):
        path2 = os.path.join(path1, content2)
        path2_resized = os.path.join(path1_resized, content2)

        print(path2_resized)

        if os.path.isfile(path2):
            copyfile(path2, path2_resized)

        if os.path.isdir(path2):
            os.makedirs(path2_resized, exist_ok=True)

            # Construct obj_train_data/PANA_112/
            for content3 in os.listdir(path2):
                path3 = os.path.join(path2, content3)
                path3_resized = os.path.join(path2_resized, content3)

                print(path3_resized)
                os.makedirs(path3_resized, exist_ok=True)

                # Construct obj_train_data/PANA_112/set1/
                for content4 in os.listdir(path3):
                    path4 = os.path.join(path3, content4)
                    path4_resized = os.path.join(path3_resized, content4)

                    print(path4_resized)
                    os.makedirs(path4_resized, exist_ok=True)

                    # Copy obj_train_data/PANA_112/set1/{img_name}.txt
                    # Resize and save obj_train_data/PANA_112/set1/{img_name}.JPG
                    for content5 in os.listdir(path4):
                        path5 = os.path.join(path4, content5)
                        path5_resized = os.path.join(path4_resized, content5)

                        print(path5_resized)
                        
                        if content5.lower().endswith(".txt"):
                            copyfile(path5, path5_resized)

                        if content5.lower().endswith(".jpg"):
                            img = cv2.imread(path5, cv2.IMREAD_UNCHANGED)
                            resized = cv2.resize(img, NEW_SIZE, interpolation = cv2.INTER_AREA)
                            cv2.imwrite(path5_resized, resized)

