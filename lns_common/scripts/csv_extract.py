import csv 
import os
import sys
import shutil

curr_dir = os.path.join(os.getcwd(), 'good_unfiltered_data')
dest_dir = os.path.join(os.getcwd(), 'good_data')

sign_types = ['pedestrianCrossing', 'stop_', 'turnLeft', 'turnRight', 'speedLimit']
old_csv_file = open('allAnnotations.csv', 'r')
full_annotations = csv.DictReader(old_csv_file, delimiter=';', fieldnames=['Filename', 'Annotation tag', 'Upper left corner X', 'Upper left corner Y',
                                  'Lower right corner X', 'Lower right corner Y', 'Occluded', 'On another road',
                                  'Origin file', 'Origin frame number', 'Origin track', 'Origin track frame number'])
new_csv_file = open('relevantAnnotations.csv', 'w')
new_annotations = csv.DictWriter(new_csv_file, fieldnames=['filename', 'annotation', 'bounding_box'])
new_annotations.writeheader()

for dirname in os.listdir(curr_dir):
    annotations_filename = dirname
    dirname = (os.path.join(curr_dir, dirname))
    if (dirname.find('.DS_Store') != -1) or (dirname.find('aiua') != -1):   #delete useless files/folders
        if os.path.isfile(dirname):
            os.remove(dirname)
        elif os.path.isdir(dirname):
            shutil.rmtree(dirname)
        continue
    annotations_filename = os.path.join(annotations_filename, os.listdir(dirname)[0])
    dirname = os.path.join(dirname, os.listdir(dirname)[0])
    if (dirname.find('.DS_Store') == -1):
        test = os.listdir(dirname)
        for filename in os.listdir(dirname):
            if (filename.find('.DS_Store') == -1):
                for type in sign_types:
                    if filename.find(type) != -1:
                        for row in full_annotations: #go into the new csv file and write that row
                            if row['Filename'] == os.path.join(annotations_filename, filename):
                                if (row['Annotation tag'] == 'speedLimit15'):
                                    folder_name = 'speedLimit15'
                                elif (row['Annotation tag'] == 'speedLimit25'):
                                    folder_name = 'speedLimit25'
                                elif type == "speedLimit":
                                    break
                                else:
                                    folder_name = type

                                new_annotations.writerow({'filename': os.path.join(annotations_filename, filename), 'annotation': row['Annotation tag'],
                                                          'bounding_box': [(row['Lower right corner X'], row['Lower right corner Y']),
                                                                           (row['Upper left corner X'], row['Upper left corner Y'])]})
                                shutil.copy(os.path.join(dirname, filename), os.path.join(os.path.join(dest_dir, folder_name),
                                                                                    filename))  # move the sign into the right folder
                                old_csv_file.seek(0)
                                break
                        old_csv_file.seek(0)
