# ai-malaria-diagnosis

AI for malaria microscopy diagnosis using a TensorFlow CNN. The repository now has a reusable source pipeline for loading images, training a baseline model, and evaluating results.

## Project Structure

```text
data/raw/
  Parasitized/
  Uninfected/
config/config.yaml
main.py
src/
  data_loader.py
  evaluate.py
  model.py
  train.py
  utils.py
notebooks/
  01_data_exploration.ipynb
  02_initial_model_experiment.ipynb
requirements.txt
requirements-dev.txt
```

## How The Files Work Together

`config/config.yaml` is the control panel. It stores data paths, image size, batch size, epochs, learning rate, output model names, and report names.

`src/utils.py` contains shared helpers for loading config, resolving project-relative paths, creating output folders, setting random seeds, and writing JSON reports.

`src/data_loader.py` reads the class folders under `data/raw`, creates training and validation datasets with `tf.keras.utils.image_dataset_from_directory`, and adds `prefetch(tf.data.AUTOTUNE)` so training can load the next batch while the model is working.

`src/model.py` defines the CNN. It includes input rescaling, light image augmentation, convolution blocks, dropout, and binary classification output.

`src/train.py` builds the datasets and model, trains with checkpointing, early stopping, and learning-rate reduction, then saves the final model and training history.

`src/evaluate.py` loads a saved model, runs validation predictions, and writes classification metrics plus a confusion matrix.

`main.py` is the command-line entry point that ties the train and evaluate steps together.

## Setup

```bash
pip install -r requirements.txt
```

For notebooks and exploration tools:

```bash
pip install -r requirements-dev.txt
```

## Run The Pipeline

Train only:

```bash
python main.py train
```

Evaluate the best checkpoint:

```bash
python main.py evaluate
```

Train and then evaluate:

```bash
python main.py all
```

Outputs are written to `models/` and `reports/`.

