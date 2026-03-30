# Project 1: Your First Neural Network
**Deep Learning Nanodegree Foundation**

This project demonstrates how to build and train a Neural Network from scratch to predict daily bike-sharing ridership. It includes both a hands-on Jupyter Notebook for manual implementation and a production-ready Python script using Scikit-Learn.

## Project Overview
By completing this project, you will understand:
- **Gradient Descent & Backpropagation:** The core algorithms that allow neural networks to learn.
- **Data Preprocessing:** Converting categorical data into dummy variables and scaling numeric features.
- **Model Evaluation:** Using metrics like MSE (Mean Squared Error) and R2 Score to measure performance.

The dataset comes from the [UCI Machine Learning Database](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset).

---

## Getting Started

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python dependency management.

### 1. Prerequisites
Install `uv` if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Run the Notebook (Manual Implementation)
The core of the Nanodegree project is implementing the neural network manually in the notebook.
```bash
uv run jupyter notebook
```
Open `dlnd-your-first-neural-network.ipynb` and follow the instructions.

### 3. Run the ML Pipeline (Scikit-Learn)
We have provided `main.py` as a modern alternative that uses `scikit-learn` to build a similar neural network (MLPRegressor). This is how you might approach the problem in a professional environment.
```bash
uv run main.py
```
This script will:
1. Load and clean the data.
2. Train a multi-layer perceptron.
3. Print performance metrics (RMSE, R2).
4. Save a visualization of the predictions to `results_test_set_predictions.png`.

---

## Project Structure
- `dlnd-your-first-neural-network.ipynb`: The main project notebook.
- `main.py`: A complete ML pipeline using Scikit-Learn.
- `Bike-Sharing-Dataset/`: Contains `hour.csv` and `day.csv`.
- `pyproject.toml`: Dependency definitions for `uv`.

## Libraries Used
- **NumPy & Pandas:** For data manipulation.
- **Matplotlib:** For visualization.
- **Scikit-Learn:** For the automated ML pipeline.
- **Jupyter:** For interactive development.
