# GSI-Performance-Prediction-MLP

This project builds upon the open-source repository GSI-Performance-Prediction by Almehedi06.[url:https://github.com/Almehedi06/GSI-Performance-Prediction]

My contribution is the addition of a Multi-Layer Perceptron (MLP) model to the existing framework.

Below is the original README file from the author for reference.


# GSI-Performance-Prediction

Welcome to my repository on **AI-based modeling for Green Stormwater Infrastructure (GSI) performance prediction**.  
This project implements a **scalable, configurable ML/DL pipeline** for evaluating and comparing multiple models under a unified structure.

## Why This Matters
Green Stormwater Infrastructure (GSI) plays a key role in **reducing urban flooding, improving water quality**, and **supporting urban ecosystems**.  
However, traditional models are **computationally expensive** and **hard to generalize**.  
This project aims to **overcome these limitations** by providing a **flexible ML/DL framework** with **model configurability** and **reproducibility**.

## Supported Models
This repository supports **training and testing** of the following models:

✅ **Random Forest (RF)** — Non-parametric ensemble learning.  
✅ **Data-Driven LSTM (LSTM)** — Sequence modeling for time-series data.  
✅ **Physics-Informed LSTM (PILSTM)** — Deep learning with physical constraints (coming soon).

Model selection is **fully controlled via `config/config.yaml`** or **CLI arguments**.

## Project Structure
```
├── config/                  # Model and data configuration (YAML)
├── data/                    # Raw and processed data (DO NOT COMMIT large files)
├── notebooks/               # Experimentation notebooks
├── results/                 # Logs, models, metrics, and plots
├── src/                     # Source code for models, utils, and pipelines
│   ├── models/              # Model definitions (RF, LSTM, PILSTM, MLP)
│   ├── utils/               # Utility functions for data and evaluation
│   ├── train.py             # Model training pipeline
│   ├── test_model.py        # Model testing pipeline
└── README.md                # Project overview (this file)
```

## Usage Example

### Train Random Forest:
```bash
python -m src.train --model rf
```

### Train LSTM:
```bash
python -m src.train --model lstm
```

