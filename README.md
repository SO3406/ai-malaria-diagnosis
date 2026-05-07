# ai-malaria-diagnosis

The Architecture and folder organization for this project is as below;

AI for Malaria Microscopy Diagnosis (CNN Project)
cnn-project-root/
│
├── data/                  # Data storage (not tracked by Git)
│   ├── raw/               # Immutable raw data (e.g., original images)
│   ├── processed/         # Cleaned/preprocessed data (e.g., resized images, numpy arrays)
│   └── external/          # Data from third-party sources
│
├── notebooks/             # Jupyter notebooks for EDA and rapid prototyping
│   ├── 01_data_exploration.ipynb
│   └── 02_initial_model_experiment.ipynb
│
├── src/                   # Source code for the project
│   ├── __init__.py
│   ├── data_loader.py     # Data loading, augmentation, and normalization
│   ├── model.py           # CNN architecture definition (e.g., PyTorch/Keras)
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation metrics and plots
│   └── utils.py           # Helper functions (e.g., logger)
│
├── models/                # Saved models (checkpoints, final models)
│   ├── best_model.h5
│   └── epoch_50.pth
│
├── config/                # Configuration files
│   └── config.yaml        # Hyperparameters (learning rate, batch size, epochs)
│
├── tests/                 # Unit and integration tests
├── .gitignore             # Files to ignore (data/, models/, .venv/)
├── README.md              # Project description and setup instructions
├── requirements.txt       # Dependencies
└── main.py                # Main entry point to run pipeline

