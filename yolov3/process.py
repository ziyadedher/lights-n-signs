
from typing import List, Dict

import os
import time

from common import config
from common.preprocess.preprocess import Preprocessor

def run_process(datasets):
	final_annotations = open("annotations.txt", "w")
	final_classes = open("classes.txt", "w")

	for dataset in datasets:
		data = Preprocessor.preprocess(dataset)

		for img, boxes in data.annotations.items():
			print("Working on image: {}              \r".format(img), end="")
			final_annotations.write("{} ".format(img))

			for box in boxes:
				final_annotations.write("{},{},{},{},{} ".format(box["x_min"], box["y_min"], box["x_max"], box["y_max"], box["class"]))

			final_annotations.write("\n")

	final_annotations.close()
	final_classes.write("\n".join(data.classes))
	final_classes.close()
	print("\n***DONE***\n")

if __name__=="__main__":
	run_process(["LISA"])
