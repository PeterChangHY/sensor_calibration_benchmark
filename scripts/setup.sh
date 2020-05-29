#!/bin/bash

set -e

virtualenv --system-site-packages .venv

source .venv/bin/activate

pip install -r requirements.txt


echo "Ok you should be good to go!"
echo "Just run the following now:"
echo "$ source .venv/bin/activate"