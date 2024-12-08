# Install dependencies.
pip install -r requirements.txt

# Download & unpack data.
mkdir -p data
cd data

kaggle competitions download -c um-game-playing-strength-of-mcts-variants
unzip um-game-playing-strength-of-mcts-variants.zip

kaggle datasets download -d jsday96/mcts-extra-training-data
unzip mcts-extra-training-data.zip

cd ..

# Train models.
mkdir -p models
python TrainCatBoost.py
python TrainLightGBM.py
python TrainTabM.py