### Yolo Keras Training

Steps to train:

1. install requirements: `pip install -r requirements.txt`
2. run setup script `bash setup.sh`
3. run process.py `python process.py`
4. (optional) run kmeans to redefine the anchors `python kmeans.py`
  This may produce negative numbers. If so run again until results are good.
5. run training: `python train.py`
6. (cleanup) `bash clean.sh`

You can change the parameters in training.py so that you can change how it
trains and how long it trains format

Similarly, change the datasets you want to you directly in process.py

convert.py can be used to convert regular tensorflow models into keras models
  and can also be used to convert traditional darknet models with weights to
  keras models. 
