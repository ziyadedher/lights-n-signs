# Lights and Signs Training

### Overview
This repository contains all code for the training and testing of all models reponsible for traffic light and sign detection in Zeus. Currently we have two main approaches to solving problems regarding the detection of traffic lights and signs: Haar cascade and deep convolutional networks contained respectively in `haar` and `net`.


### Details
#### Haar
Haar cascades are currently our top-of-the-line approach for solving the issue of finding and recognizing stop signs. We plan on extending this to all signs. Haar cascades are particularly suited for this problem because the inherent "feature-finding" of the cascade training process lends itself to detecting the very contrasting features of traffic signs that are designed to be very visible to drivers. In addition, Haar cascades are a very efficient solution to this problem, both processing- and storage-wise.

More information is available in the `README.md` inside of the `haar` folder.

#### Networks
Deep convolutional neural networks are another technique we are exploring, mostly for the detection of traffic lights due to their more variable and less feature-constrained looks. We are currently exploring using pre-trained networks and shifting them to be more targeted towards our particular use-case. Although slower, the deep neural networks have until now proven to be better suited for traffic light detection.

One example of such network frameworks that we are using is the recently unveiled Yolo v3; its paper can be found [here](https://pjreddie.com/media/files/papers/YOLOv3.pdf).


More information is available in the `README.md` inside of the `net` folder.


### Vision
The vision for this repository is ultimately simple: we would like to devise a system to _very_ accurately recognize traffic light state and many different types of traffic signs given an image in order to be utilized by other teams controlling the movement of Zeus.

We also really would not like Zeus to get a traffic violation.
