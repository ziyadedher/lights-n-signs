#!/bin/sh

# Space-separated package dependencies
DEPS="curl unzip"

# Check for unmet dependencies
unmet=0
for dep in ${DEPS}; do
  if ! [ -x "$(command -v ${dep})" ]; then
    echo "Error: ${dep} is not installed." >&2;
    unmet=$((unmet + 1));
  fi
done
if ! [ "${unmet}" -eq 0 ]; then
  echo "${unmet} unmet dependencies.";
  exit 1;
fi

# Download files
curl http://cvrr.ucsd.edu/vivachallenge/data/Lights_Detection/LISA_TL_dayTrain.zip --output LISA_TL_dayTrain.zip
curl http://cvrr.ucsd.edu/vivachallenge/data/Lights_Detection/LISA_TL_nightTrain.zip --output LISA_TL_nightTrain.zip
curl http://cvrr.ucsd.edu/vivachallenge/data/Lights_Detection/LISA_TL_dayTest.zip --output LISA_TL_dayTest.zip
curl http://cvrr.ucsd.edu/vivachallenge/data/Lights_Detection/LISA_TL_nightTest.zip --output LISA_TL_nightTest.zip

# Unzip files
unzip LISA_TL_dayTrain.zip
unzip LISA_TL_nightTrain.zip
unzip LISA_TL_dayTest.zip
unzip LISA_TL_nightTest.zip
