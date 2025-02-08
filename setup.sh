#!/usr/bin/env bash

echo "Installing pip dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Installing spaCy model..."
python -m spacy download en_core_web_sm

echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)"
