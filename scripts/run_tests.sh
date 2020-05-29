#!/bin/bash

echo "--------------"
echo "Style checking"
echo "--------------"
flake8 .

echo "------------"
echo "Unit testing"
echo "------------"
python -m pytest