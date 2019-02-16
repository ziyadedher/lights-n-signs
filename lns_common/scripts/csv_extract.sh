wget http://cvrr.ucsd.edu/LISA/Datasets/signDatabasePublicFramesOnly.zip
unzip signDatabasePublicFramesOnly.zip

cd signDatabsePublicFramesOnly
mkdir LISA_signs
cd LISA_signs
mkdir pedestrianCrossing
mkdir speedLimit15
mkdir speedLimit25
mkdir stop_
mkdir turnLeft
mkdir turnRight
cd ..
mv vid0 good_unfiltered_data/vid0
mv vid1 good_unfiltered_data/vid1
mv vid2 good_unfiltered_data/vid2
mv vid3 good_unfiltered_data/vid3
mv vid4 good_unfiltered_data/vid4
mv vid5 good_unfiltered_data/vid5
mv vid6 good_unfiltered_data/vid6
mv vid7 good_unfiltered_data/vid7
mv vid8 good_unfiltered_data/vid8
mv vid9 good_unfiltered_data/vid9
mv vid10 good_unfiltered_data/vid10
mv vid11 good_unfiltered_data/vid11
cd ..
mv csv_extract.py signDatabsePublicFramesOnly/csv_extract.py
cd signDatabsePublicFramesOnly

python csv_extract.py

mv relevantAnnotations.csv LISA_signs/relevantAnnotations.csv