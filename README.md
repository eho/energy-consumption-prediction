<h1 align="center">
  Machine Learning Project Template
  <br />
  <a href="https://github.com/eho/ml-project-template/issues">
    <img
      alt="GitHub issues"
      src="https://img.shields.io/github/issues/eho/ml-project-template?logo=git&style=plastic"
    />
  </a>
  <a href="https://github.com/eho/ml-project-template/network">
    <img
      alt="GitHub forks"
      src="https://img.shields.io/github/forks/eho/ml-project-template?style=plastic&logo=github"
    />
  </a>
  <a href="https://github.com/eho/ml-project-template/stargazers">
    <img
      alt="GitHub stars"
      src="https://img.shields.io/github/stars/eho/ml-project-template?style=plastic&logo=github"
    />
  </a>
</h1>

## Objective

This template provides a basic directory structure to start a machine learning/deep learning-based project.

```
project_root/
│
├── data/                     # Directory to store datasets
│   ├── raw/                  # Raw data
│   └── processed/            # Preprocessed data
│
├── src/                      # Source code for the project
│   ├── __init__.py           # Mark directory as a Python package
│   ├── dataset.py            # Custom dataset class
│   ├── model.py              # Model architecture
│   ├── train.py              # Training loop
│   ├── eval.py               # Evaluation script
│   └── utils.py              # Utility functions (logging, metrics, etc.)
│
├── experiments/              # Store experiment configurations and results
│   ├── logs/                 # Logs from experiments
│   └── checkpoints/          # Model checkpoints
│
├── config/                   # Configuration files for experiments
│   ├── default_config.yaml   # Default config
│   └── experiment_config.yaml # Specific experiment config
│
├── scripts/                  # Scripts for running experiments, data preprocessing, etc.
│   ├── preprocess_data.py    # Data preprocessing script
│   └── run_training.sh       # Bash script to run training
│
├── tests/                    # Unit tests
│   └── test_model.py         # Tests for model and utils
│
├── requirements.txt          # List of project dependencies
├── README.md                 # Project overview and instructions
├── .gitignore                # Ignoring unnecessary files for Git
└── main.py                   # Entry point for training or evaluation```

```

