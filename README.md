# Electric Vehicle Price Prediction

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
5. [Results](#results)
6. [Dependencies](#dependencies)
7. [How to Run](#how-to-run)
8. [Future Improvements](#future-improvements)

## Project Overview

This project aims to predict the prices of electric vehicles (EVs) based on various features, such as vehicle specifications, utility, and age. By leveraging data preprocessing techniques and machine learning models, the project achieves an accuracy score of 85.7%%, showcasing its effectiveness in predicting EV prices.

## Features

- Data preprocessing, including handling missing values, encoding categorical variables, and scaling features.
- Implementation of machine learning algorithms, such as Decision Trees, Support Vector Regression (SVR) and Linear Regression.
- Model evaluation using metrics like R-squared and RMSE.
- Hyperparameter tuning with GridSearchCV incorporating cross validation to optimize model performance.

## Dataset

The dataset includes information about electric vehicles, such as:
- **Model Year**
- **Make**
- **Electric Utility**
- **CAFV Eligibility**
- **Vehicle Range**
- **Expected Price**
Dataset can be found [here](https://drive.google.com/file/d/1kZ299dY3rKLvvnfTsaPtfrUbZb7k31of/view)

The data was cleaned and processed to ensure readiness for training and testing machine learning models.

## Methodology

1. **Data Cleaning**: Handled missing values using backward filling.
2. **Feature Engineering**: 
   - Encoded categorical features with Label Encoding and Target Encoding.
   - Computed the vehicle's age based on the model year.
3. **Exploratory Data Analysis**:
   - Visualized data distributions and feature correlations.
4. **Model Training**:
   - Split data into training and testing sets.
   - Standardized the features for better performance.
   - Trained multiple models, including Decision Trees, Linear Regression and SVR.
5. **Model Evaluation**:
   - Compared model performance using evaluation metrics.
   - Decision Tree Regressor performed best.

## Results

- **Best Model**: Decision Tree Regressor
- **Accuracy**: 85% (training on 80% of the dataset)
- **R-squared**: High predictive power on the testing set.

## Dependencies

This project requires the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

## How to Run

1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```
3. Open the Jupyter Notebook and run the cells sequentially.

## Future Improvements

- Experiment with other advanced algorithms, such as Random Forests or Gradient Boosting Machines.
- Perform hyperparameter optimization to further enhance model accuracy.
- Include more features for better predictions.
