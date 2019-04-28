#!/usr/bin/env bash

pip install -r ./requirements.txt

python -m spacy download en

python ./main_crawler.py
python ./main_architecture.py