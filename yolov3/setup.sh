wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5

wget https://pjreddie.com/media/files/darknet53.conv.74
mv darknet53.conv.74 darknet53.weights
python convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5

