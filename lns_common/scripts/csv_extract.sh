wget http://cvrr.ucsd.edu/LISA/Datasets/signDatabasePublicFramesOnly.zip
unzip signDatabasePublicFramesOnly.zip

cd signDatabsePublicFramesOnly
mkdir KITTI
cd KITTI
mkdir pedestrianCrossing
mkdir speedLimit15
mkdir speedLimit25
mkdir stop_
mkdir turnLeft
mkdir turnRight
cd ..
mv vid0 positives good_unfilterred_data/vid0
mv vid1 positives good_unfilterred_data/vid1
mv vid2 positives good_unfilterred_data/vid2
mv vid3 positives good_unfilterred_data/vid3
mv vid4 positives good_unfilterred_data/vid4
mv vid5 positives good_unfilterred_data/vid5
mv vid6 positives good_unfilterred_data/vid6
mv vid7 positives good_unfilterred_data/vid7
mv vid8 positives good_unfilterred_data/vid8
mv vid9 positives good_unfilterred_data/vid9
mv vid10 positives good_unfilterred_data/vid10
mv vid11 positives good_unfilterred_data/vid11
cd ..
mv csv_extract.py signDatabsePublicFramesOnly/csv_extract.py
cd signDatabsePublicFramesOnly

python csv_extract.py

mv relevantAnnotations.csv KITTI/relevantAnnotations.csv