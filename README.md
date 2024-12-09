# Overview

This repo contains the training, test, and data generation infrastructure of the 1st place solution to the [Game-Playing Strength of MCTS Variants Kaggle competition](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/).

# Reproducing the winning ensemble

The models used in the winning solution can be replicated by running
```
./TrainWinningEnsemble.sh
```
in the root of the repository. This will install dependencies, download training data, train the models, and save the models to a `models` subdirectory of the current working directory. You will need an NVIDIA GPU with at least 8 GB of VRAM and at least 16 GB of system RAM to run it.

# Data generation/annotation
* Text generation models for this repo's GAVEL implementation are available at [https://www.kaggle.com/datasets/jsday96/gavel-models](https://www.kaggle.com/datasets/jsday96/gavel-models)
* The main entrypoint for running GAVEL is `GAVEL/GenerateGames.py`
* A utility for generating rulesets with ordinary instruction-tuned LLMs and few-shot prompting is located at `DataGeneration/NewGameGeneration/GenerateNewGames.py`
* Data annotation utilities are scattered throughout the `DataGeneration`, `GAVEL`, `StartingPositionEvaluation`, and `Reannotation` directories.

# Starting position analysis
Source code for the utility which runs tree searches on game starting positions to compute additional game balance & search speed features is located at `StartingPositionEvaluation/EclipseWorkspace/AnalyzeGameStartingPosition/src/analyzeGameStartingPosition/AnalyzeGameStartingPosition.java`. It depends on the [Ludii player](https://github.com/Ludeme/Ludii).