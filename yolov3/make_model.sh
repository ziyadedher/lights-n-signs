mkdir yolo_model
cp -r yolo3 yolo_model
cp yolo.py yolo_model
mkdir yolo_model/logs
mkdir yolo_model/logs/000/
cp logs/000/trained_weights_final.h5 model/logs/000/trained_weights_final.h5
cp classes.txt yolo_model
mkdir yolo_model/model_data
cp model_data/yolo_anchors.txt yolo_model/model_data/yolo_anchors.txt
mkdir yolo_model/lns_common
cp lns_common/model.py yolo_model/lns_common/model.py
touch yolo_model/__init__.py
cp requirements.txt yolo_model/requirements.txt
