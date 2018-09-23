#!/usr/bin/env bash


directory=$(mktemp -d)
sitepackages=$(python -c "import site; print(site.getsitepackages()[0])")
packagename="squeezedet_keras"
packagepath=$sitepackages/$packagename

cd $directory
git clone https://github.com/omni-us/squeezedet-keras.git
cd squeezedet-keras

while true; do
    read -p "Do you want to delete $packagepath [y/n]? " yn
    case $yn in
        [Yy]* ) rm -rf $packagepath; break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

mkdir $packagepath
cp -r main/* $packagepath/
echo "Configuration completed."
