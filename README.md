# California Housing Price Prediction using XGBoost

## Project Overview

This project builds a machine learning model to predict **median house values** in California districts using the **XGBoost Regressor**. The dataset is sourced from `sklearn.datasets` and includes various socio-economic and geographic features.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost

---

## Dataset Information

The dataset used is the **California Housing Dataset**, which contains **20,640 rows** and **9 columns**.

### Features:

* `MedInc` → Median income in block group
* `HouseAge` → Median house age
* `AveRooms` → Average number of rooms
* `AveBedrms` → Average number of bedrooms
* `Population` → Block population
* `AveOccup` → Average occupancy
* `Latitude` → Latitude location
* `Longitude` → Longitude location

### Target:

* `MedHouseVal` → Median house value

---

## Data Exploration

* Checked for missing values → ✅ None found
* Generated statistical summary using `.describe()`
* Visualized feature correlations using a heatmap

---

## Feature Correlation

A heatmap was used to understand relationships between variables. This helps identify:

* Strong predictors
* Multicollinearity

---

## Data Splitting

The dataset was split into:

* **Training Set (80%)**
* **Testing Set (20%)**

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
```

---

## Model Used: XGBoost Regressor

The model used is:

```python
from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(X_train, Y_train)
```

### Why XGBoost?

* High performance and accuracy
* Handles non-linear relationships
* Works well with structured/tabular data
* Automatically handles missing values

---

## Model Evaluation

### Training Data Performance

* **R² Score:** 0.9436
* **Mean Absolute Error (MAE):** 0.1933

### Test Data Performance

* **R² Score:** 0.8338
* **Mean Absolute Error (MAE):** 0.3108

---

## Visualization

A scatter plot was used to compare:

* Actual Prices vs Predicted Prices

```python
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
```

---

## Key Insights

* The model performs very well on training data (high R²)
* Slight drop in test performance indicates **minor overfitting**
* XGBoost is highly effective for regression problems with structured data

---

## How to Run the Project

1. Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

2. Run the script or Jupyter Notebook

3. Train the model and evaluate results

---

## Future Improvements

* Hyperparameter tuning (GridSearchCV)
* Feature engineering
* Cross-validation
* Model comparison (Linear Regression, Random Forest)

