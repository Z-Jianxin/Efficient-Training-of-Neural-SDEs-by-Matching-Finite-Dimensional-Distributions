# Repository

This repository implements the FDM training method for NeuralSDE proposed in

```
@inproceedings{
zhang2025efficient,
title={Efficient Training of Neural Stochastic Differential Equations by Matching Finite Dimensional Distributions},
author={Jianxin Zhang and Josh Viktorov and Doosan Jung and Emily Pitler},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=d4qMoUSMLT}
}
```

## Repository Structure

```
├── train_unconditional.py         # Main script for training the model
├── config.py                      # Configuration file for model, data, and hyperparameters
├── test_forex_metals64.ipynb      # Jupyter notebook for testing and demonstration
├── src/                           # Source code for model components and utilities
├── data/                          # Directory for datasets
├── requirements.txt               # Python dependencies
├── README.md                      # This README file
```

## Prerequisites

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Training

To train the NeuralSDE model, run the following command:
```bash
python train_unconditional.py --config_file config.py
```

The `config.py` file contains all necessary configurations, including:
- Model architecture
- Dataset paths
- Hyperparameters

Modify the `config.py` file to adjust these parameters as needed.

## Testing

Demonstrate testing and evaluation of the trained model using the provided Jupyter notebook:
```bash
test_forex_metals64.ipynb
```

Follow the instructions in the notebook to load the trained model and visualize the results.

## Data

Place your datasets in the `data/` directory. Update the paths in the `config.py` file accordingly.

## Acknowledgments
Our code is based on the code release of the paper Non-adversarial training of Neural SDEs with signature kernel scores on https://github.com/issaz/sigker-nsdes/.

