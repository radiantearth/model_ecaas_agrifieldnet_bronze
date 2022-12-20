#!/usr/bin/env bash

# set -e

echo "Inferring"

python d013i.py
python d013i-support.py

python pre_tif2png.py
python pre_features.py

python m013-37i.py

echo "Done"

bash