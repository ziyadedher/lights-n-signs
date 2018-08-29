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


# TODO: somehow link this to the resources data root
RESOURCES_ROOT="${~/.lns-training/resources/#\~$HOME}"

# Create main directories
mkdir -p $RESOURCES_ROOT
cd $RESOURCES_ROOT
mkdir -p raw
mkdir -p data


# === LISA ===
cd raw
mkdir -p LISA
cd LISA
curl http://cvrr.ucsd.edu/vivachallenge/data/Lights_Detection/LISA_TL_dayTrain.zip --output LISA_TL_dayTrain.zip

cd ../..
cd data
mkdir -p LISA
cd LISA
unzip LISA_TL_dayTrain.zip -d .
