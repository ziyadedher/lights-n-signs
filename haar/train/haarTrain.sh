opencv_traincascade -data data -vec positives.vec -bg $1_neg.txt -numPos 3500 -numNeg 2000 -numStages $3 -w $2 -h $2
