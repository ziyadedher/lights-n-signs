mkdir model
cp -r yolo3 model
cp yolo.py model
mkdir model/logs
mkdir model/logs/000/
cp logs/000/trained_weights_final.h5 model/logs/000/trained_weights_final.h5
cp classes.txt model
mkdir model/model_data
cp model_data/yolo_anchors.txt model/model_data/yolo_anchors.txt
mkdir model/lns_common
cp lns_common/model.py model/lns_common/model.py
touch model/__init__.py

