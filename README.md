# Traffic Lights and Signs Training

### Overview
This repository contains all code for the training the of models that are responsible for all traffic light and sign detection in Zeus.

### Workflow
#### Data
Before anything can happen, the required data must be downloaded. The `data` subdirectory has a cross-platform script that will download and unpack our standard dataset for training. Be aware that this dataset is very large (22.5GB+ when unpacked).

#### Preprocessing
Before training can begin, the downloaded data must be preprocessed. The `preprocess` subdirectory has a script to do all the data preprocessing and generate the annotations required. The data annotations are formatted as in a regular Haar cascade annotation file.

Changes can be made to what exactly is being trained on inside of the preprocessing script.

#### Training
OpenCV is used for all Haar cascade training, so make sure that is installed before attempting any of the subsequent steps. In addition, the Python 3.6 bindings of OpenCV are required for everything to function normally.

Firstly, setup the dataset for training on a specific type of light by calling `trainSetup.sh` with first argument the type of light to train on and second argument the size of the features. The type of lights currently supported are `Green`, `Red`, `Yellow`, `GreenLeft`, `RedLeft`, and `YellowLeft`.

Then, run the the training script by calling `haarTrain.sh` with with first argument the type of light to train on, second argument the size of the features, and third argument the number of stages to train for.

A general tutorial for Haar cascade in training in OpenCV using Python can be found [here](https://pythonprogramming.net/haar-cascade-object-detection-python-opencv-tutorial/).

The full `cascade.xml` file is only saved at the end of training for the number of stages you specified; however, snapshots are saved for every stage.

#### Testing
After training the cascade and aquiring a cascade file, it can be tested out. There are two types of testing currently available: live video testing and testing using the testing set. All testing is under the `testHaar` subdirectory. Neighbours and scaling can be changed from inside of each testing file.

##### Live Video Testing
To test using a live video feed from a webcam run `videoTest.py`. Bounding boxes will be displayed on the live feed.

##### Testing Set
To test using a testing set and validate the cascade with statistics run `fileTest.py` with first argument the path to the cascade file, second argument the negative annotations file, and third argument the positive annotations file.

This file can also visualize the bounding boxes on the images; this can be enabled from inside the file.

#### Quickstart
Here is a quickstart guide for getting this repository running; make sure to change the things between angled bracket (`<`, `>`) with their respective values.
  1. Install OpenCV with Python 3.6 bindings.
  1. Clone repository: `git clone git@gitlab.com:aUToronto/autonomy/lights-n-signs-training.git`.
  1. Get data: `cd lights-n-signs-training/data && ./download.sh`.
  1. Preprocess data: `cd ../preprocess && python3 processData.py`.
  1. Setup for training `cd ../haar && ./trainSetup.sh <LIGHT_TYPE> <FEATURE_SIZE>`.
  1. Train cascade: `./trainSetup.sh <LIGHT_TYPE> <FEATURE_SIZE> <NUM_STAGES>`.
  1. Test cascade: `cd ../testHaar`
    - Live video test: `python3 videoTest.py`.
    - File test: `python3 fileTest.py <CASCADE_PATH> <NEGATIVE_ANNOTATIONS_PATH> <POSITIVE_ANNOTATIONS_PATH>`.
  1. Rejoice.
