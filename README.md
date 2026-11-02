# Fish Weight Prediction Using Polynomial Regression

## Project Overview

This project aims to predict fish weight based on physical measurements using **Polynomial Regression** implemented with PyTorch.

The main objective is to investigate and compare five optimization approaches and evaluate their performance in minimizing prediction error.

## Dataset

The project uses the **Fish Market Dataset**, available on Kaggle.

**Dataset Link:** https://www.kaggle.com/datasets/vipullrathod/fish-market

The original dataset contains 159 records and 7 columns.

**Input Features:**
- Length1
- Length2
- Length3
- Height
- Width

**Target Variable:**
- Weight (grams)

The `Species` column is excluded from the model's input features.

## Data Preprocessing

The following preprocessing steps are applied:

1. Remove records with zero or negative fish weight.
2. Split the dataset into 80% training and 20% testing.
3. Apply Polynomial Features with degree 2.
4. Standardize input features using StandardScaler.
5. Standardize the target variable for stable training.
6. Convert the processed data into PyTorch tensors.

## Model Architecture

The project uses a polynomial regression model implemented with PyTorch.

- **Polynomial Degree:** 2
- **Input Features After Transformation:** 20
- **Model:** Single Linear Layer (`nn.Linear(20, 1)`)
- **Loss Function:** Mean Squared Error (MSE)
- **Training Epochs:** 200

## Optimization Algorithms

Five optimization approaches are implemented and evaluated:

| Optimizer | Description |
|---|---|
| Gradient Descent (GD) | Uses the full training dataset for each parameter update |
| Stochastic Gradient Descent (SGD) | Updates parameters after each training sample |
| Mini-Batch Gradient Descent | Updates parameters using batches of 16 samples |
| RMSprop | Uses adaptive learning rates based on moving averages of squared gradients |
| Adam | Combines adaptive learning rates with first-moment gradient estimates |

Momentum is incorporated into the optimization configurations where supported.

## Technologies and Libraries

- Python
- Kaggle Notebooks
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- GitHub

## Evaluation Metrics

The models are evaluated using:

- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **R² Score:** Coefficient of Determination

Predictions are converted back to the original weight scale before evaluation.

Training loss curves and actual-versus-predicted plots are used to visualize model performance.


## Project Objective

The project investigates how different optimization algorithms affect:

- Model convergence
- Training stability
- Prediction accuracy
- Optimization efficiency


