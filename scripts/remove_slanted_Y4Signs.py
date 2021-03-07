import cv2
import os

from shutil import copyfile

"""
Need to remove images containing any slanted signs from Y4Signs
"""

ROOT = "/mnt/ssd6/Y4Signs"
ROOT_REMOVED_SLANTED = f"/mnt/ssd1/lns/resources/data/Y4Signs_removed_slanted"

IMG_WIDTH = 5184
IMG_HEIGHT = 2920

# Looked at images manually and saw that the thresholds below are reasonable
# The Stop sign has a dummy value since we don't care about that sign
sign2threshold = {0: 0.75, # nrt text
                1: 0.88, # nrt sym
                2: 0.75, # nlt text
                3: 0.88, # nlt sym
                4: 0.85, # yield
                5: -1} # stop

sign2count_removed = {}

# Construct set01, ..., set62
for content1 in os.listdir(ROOT):
    path1 = os.path.join(ROOT, content1)
    path1_removed_slanted = os.path.join(ROOT_REMOVED_SLANTED, content1)
    print(path1_removed_slanted)

    os.makedirs(path1_removed_slanted, exist_ok=True)
    
    # Construct obj_train_data/, obj.data, obj.names, train.txt
    for content2 in os.listdir(path1):
        path2 = os.path.join(path1, content2)
        path2_removed_slanted = os.path.join(path1_removed_slanted, content2)

        print(path2_removed_slanted)

        if os.path.isfile(path2):
            copyfile(path2, path2_removed_slanted)

        if os.path.isdir(path2):
            os.makedirs(path2_removed_slanted, exist_ok=True)

            # Construct obj_train_data/PANA_112/
            for content3 in os.listdir(path2):
                path3 = os.path.join(path2, content3)
                path3_removed_slanted = os.path.join(path2_removed_slanted, content3)

                print(path3_removed_slanted)
                os.makedirs(path3_removed_slanted, exist_ok=True)

                # Construct obj_train_data/PANA_112/set1/
                for content4 in os.listdir(path3):
                    path4 = os.path.join(path3, content4)
                    path4_removed_slanted = os.path.join(path3_removed_slanted, content4)

                    print(path4_removed_slanted)
                    os.makedirs(path4_removed_slanted, exist_ok=True)

                    # Copy obj_train_data/PANA_112/set1/{img_name}.txt
                    # Resize and save obj_train_data/PANA_112/set1/{img_name}.JPG
                    for content5 in os.listdir(path4):
                        path5 = os.path.join(path4, content5)
                        path5_removed_slanted = os.path.join(path4_removed_slanted, content5)

                        
                        if content5.lower().endswith(".txt"):
                            # Read the sign, width and height
                            # If the aspect ratio for any bbox is too small, ignore the image and txt.
                            # Otherwise, copy both the image and the txt.
                            f = open(path5, "r")
                            lines = f.readlines()
                            for line in lines:
                                line_content = line.split(" ")
                                sign = int(line_content[0])
                                w = float(line_content[3]) * IMG_WIDTH
                                h = float(line_content[4]) * IMG_HEIGHT
                                aspect_ratio = w / h

                                print(f"{path5_removed_slanted} | sign {sign} | aspect ratio {aspect_ratio}")

                                if aspect_ratio < sign2threshold[sign]:
                                    # Keep track of how many we remove
                                    if sign not in sign2count_removed:
                                        sign2count_removed[sign] = 0
                                    sign2count_removed[sign] += 1

                                    break
                            else:
                                print("Copying")
                                copyfile(path5, path5_removed_slanted)

                                src_img_name = path5[:-4] + ".JPG"
                                if os.path.exists(src_img_name):
                                    dst_img_name = path5_removed_slanted[:-4] + ".JPG"
                                    copyfile(src_img_name, dst_img_name)
                                else:
                                    src_img_name = path5[:-4] + ".jpg"
                                    dst_img_name = path5_removed_slanted[:-4] + ".jpg"
                                    copyfile(src_img_name, dst_img_name)

print(f"Signs removed: {sign2count_removed}")
