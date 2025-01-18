# Real Estate Price Prediction

This repository contains a Python project that predicts home prices based on various features such as location, size, number of bathrooms, and bedrooms using machine learning models. The goal is to help estimate the price of properties in Bangalore based on historical listings.

## Files Overview

### 1. **Data Preprocessing**

The code begins by importing necessary libraries and reading the real estate dataset, followed by a series of preprocessing steps:

- **Histograph of price_per_sqft**: A histogram is plotted to understand the distribution of price per square foot in the dataset.
  
- **Bathroom Column Analysis**: Analyzes the number of bathrooms in the dataset and filters out listings with excessive bathroom counts compared to bedrooms.

- **Feature Engineering**:
  - **One-Hot Encoding**: The `location` column is one-hot encoded to create dummy variables for each unique location in the dataset.
  - **Data Clean-Up**: Unnecessary columns are dropped, and the dataset is prepared for machine learning modeling.

### 2. **Model Training and Evaluation**

- **Linear Regression**: The dataset is split into training and testing sets, and a Linear Regression model is trained to predict home prices.
  
- **Cross-Validation**: The model is evaluated using K-fold cross-validation to assess its generalization ability across different subsets of the dataset.
  
- **GridSearchCV**: A grid search is used to find the best hyperparameters for the Linear Regression, Lasso, and Decision Tree models.

- **Standard Scaling**: Features are scaled using StandardScaler to improve model performance.

### 3. **Model Selection**

The best performing model is chosen based on cross-validation scores, and it is confirmed that **Linear Regression** yields the best results for this dataset.

### 4. **Price Prediction**

A custom function is implemented to predict the price of a property based on inputs like location, square footage, number of bathrooms, and bedrooms.

```python
def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return lr_clf.predict([x])[0]


# Bangalore Home Prices Prediction

This project uses machine learning techniques to predict the price of homes in Bangalore based on various features such as the number of bedrooms, bathrooms, square footage, and location.

## Requirements

- Python 3.x
- Pandas
- Matplotlib
- NumPy
- scikit-learn
- pickle

To install the required packages, run:

```bash
pip install -r requirements.txt
