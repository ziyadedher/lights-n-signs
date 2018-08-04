#!/bin/sh

# Configure PYTHONPATH to include the directory this script is in
export PYTHONPATH=$PYTHONPATH:$(dirname $(readlink -f "$0"))
