This will be the folder that all traffic light and sign detection happens in.

The data folder has a script which will download and unpack all necessary data.

Be sure to install the python version of opencv before you try and do anything else.

The data annotations are formatted like seen in a haar cascade annotation file.
This means that they take on the following format in the positive annotation file (pos.txt):
  - filespath numberOfInstancesInPhoto x y width height ...

To run the haar cascade, first run the trainSetup.sh with the first argument being the
type of light you want to train for out of the following options:
  -Green
  -Red
  -Yellow
  -GreenLeft
  -RedLeft
  -YellowLeft
And the second argument being the size of the features (20 by 20 is standard)

Then run the haarTrain.sh with the type of light as the first argument, and the size of
the features in the second layer and number of stages as the third (10-20 is standard)

Link to a general tutorial can be found here:
https://pythonprogramming.net/haar-cascade-object-detection-python-opencv-tutorial/

NOTE:
  The number of samples we can work will is limited to just below 6000. The current settings
  should ensure that this is not a problem but modify the haarTrain.sh and the trainSetup.sh
  if you want these values to be different.

The cascades are saved only at the end of the training but a snapshot is saved at the end of
each stage. The snapshots and final cascade can be found in the data folder.

The video test file will automatically test the detection on your computer camera video feed.
Change the running parameters (neighbours and scale) in the file

NOTE#2:
  -in the preprocessing file you can change which types of lights you want to detect, and whether
    you work with the greyscale of the entire image or just focus on one color of pixel. This change
    help to detect traffic lights of a particular color

fileTest.py will validate the data after the test. It follows the following format
  fileTest.py cascadepath.xml negFiles.txt posFiles.txt

It also contains a file to visualize the bounding boxes on the images. Uncomment the commented
  Line in the main section to run it. This will help validate the data itself.

Have fun!
