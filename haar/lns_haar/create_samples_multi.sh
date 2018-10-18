#!/usr/bin/env bash

input=$1
output=$2
num_per_sample=$3
bg_file=$4

counter=0

for f in $input/*
do
	echo "Creating $num_per_sample samples for file $f..."
	opencv_createsamples -img $f -vec $output/file_$counter.vec -bg $bg_file -num $num_per_sample
	counter=`expr $counter + 1`
done

exit 0
