# 🎯 MACHINE LEARNING MIDTERM EXAM REVIEWER

### Based on: Prof's Notebooks (Regression_Demo + Tuning) + Lab 4 (Kaggle Competition)

---

## 📋 MIDTERM EXAM STRUCTURE

1. Exploratory Data Analysis (same flow as prelim, but dataset is **regression**)
2. Data Preprocessing — including regression-specific steps (log transforms, skew handling)
3. Model Training — at least **3 Regression algorithms**
4. Hyperparameter Tuning (GridSearchCV / RandomizedSearchCV / Optuna)
5. Model Comparison & Evaluation using **Regression metrics**

> ⚠️ KEY DIFFERENCES FROM PRELIM:
>
> - Prelim = **Classification** → predict a label → metrics: Accuracy, F1, Confusion Matrix
> - Midterm = **Regression** → predict a **number** → metrics: RMSE, MAE, R², RMSLE
> - Midterm adds: **Cross-Validation**, **Hyperparameter Tuning**, **Ensemble methods**
> - No `stratify=y` in train_test_split for regression!

---

# ⚡ QUICK START — COPY THIS AT THE TOP OF YOUR NOTEBOOK

```python
# ── DATA & MATH ──────────────────────────────────────────
import pandas as pd
import numpy as np

# ── VISUALIZATION ─────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns

# ── PREPROCESSING ─────────────────────────────────────────
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OrdinalEncoder

# ── REGRESSION MODELS ─────────────────────────────────────
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
# Gradient boosting powerhouses (install if not present):
# pip install xgboost lightgbm catboost
import xgboost as xgb                          # XGBRegressor
from lightgbm import LGBMRegressor             # LightGBM
# from catboost import CatBoostRegressor       # CatBoost (optional)

# ── REGRESSION METRICS ────────────────────────────────────
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── SETTINGS ──────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
```

---

# PHASE 1: EDA (Same as Prelim — Quick Recap)

## 1.1 DATA INSPECTION

```python
df = pd.read_csv('your_dataset.csv')

df.head()           # first 5 rows
df.tail()           # last 5 rows
df.info()           # dtypes, non-null counts
df.describe()       # mean, std, min, max, quartiles
df.shape            # (rows, cols)
df.dtypes
df.nunique()
df.isnull().sum()
```

**Write an insight like:**

> "The dataset has X rows and Y columns. The target variable is [column name] which is a **continuous numeric variable**, making this a **regression** problem."

---

## 1.2 DATA CLEANING

```python
# Check and remove duplicates
df.drop_duplicates(inplace=True)

# Missing values — how many and what % ?
df.isnull().sum()
df.isnull().sum() / len(df) * 100

# Decision:
# < 5% missing  → drop those rows
# 5–30% missing → fill (mean for normal, median for skewed)
# > 30% missing → drop the whole column

df.dropna(inplace=True)                                  # drop rows
df['col'].fillna(df['col'].median(), inplace=True)       # fill numeric
df['col'].fillna(df['col'].mode()[0], inplace=True)      # fill categorical

# Fix wrong data types
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['date']  = pd.to_datetime(df['date'])

# Strip extra whitespace from column names and string values
df.columns = df.columns.str.strip()
df['text_col'] = df['text_col'].str.strip().str.lower()
```

### 🔑 Regression-Specific Cleaning Tips (from Lab 4)

**Remove zero or negative values if target is a price/count:**

```python
df = df[df['target_col'] > 0].copy()      # prices can't be 0 or negative
df['quantity'] = df['quantity'].clip(lower=0)   # quantities: clip, don't remove
```

**Safe bulk null-fill using column medians (very useful in exams):**

```python
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())
```

**After train/test split, fill test nulls with TRAIN medians to avoid leakage:**

```python
for col in X_train.columns:
    med = X_train[col].median()           # always from TRAIN only
    X_train[col] = X_train[col].fillna(med)
    X_test[col]  = X_test[col].fillna(med)
```

---

## 1.3 OUTLIER DETECTION

```python
# IQR Method — use for numeric columns
Q1  = df['col'].quantile(0.25)
Q3  = df['col'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Check how many outliers
outliers = df[(df['col'] < lower) | (df['col'] > upper)]
print(f"Outliers in col: {len(outliers)}")

# Remove them (optional — justify in your insight!)
df = df[(df['col'] >= lower) & (df['col'] <= upper)]
```

**Boxplot to visualize outliers:**

```python
numerical_cols = df.select_dtypes(include=np.number).columns
fig, axes = plt.subplots(1, len(numerical_cols), figsize=(16, 5))
for i, col in enumerate(numerical_cols):
    axes[i].boxplot(df[col].dropna())
    axes[i].set_title(col)
plt.tight_layout()
plt.show()
```

---

## 1.4 FEATURE ENGINEERING

```python
# Create ratio/combination features
df['price_per_sqft'] = df['price'] / df['area']
df['total_assets']   = df['col1'] + df['col2'] + df['col3']

# Bin continuous variable
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100],
                          labels=['Child', 'Young Adult', 'Adult', 'Senior'])

# Extract from date
df['date']        = pd.to_datetime(df['date'])
df['year']        = df['date'].dt.year
df['month']       = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek

# Drop useless columns (IDs, names)
df.drop(['id', 'name'], axis=1, inplace=True)
```

### 🔑 Regression-Specific Feature Engineering (from Lab 4)

**Log-transform highly skewed numeric features** (not just the target!):

```python
# Check skewness first
for col in df.select_dtypes(include=np.number).columns:
    skew = df[col].skew()
    if abs(skew) > 1.5 and df[col].min() >= 0:
        df[col] = np.log1p(df[col])   # log1p = log(x+1), safe when x=0
        print(f"Log-transformed: {col} (was skew={skew:.2f})")
```

**Winsorization — clip extreme outliers instead of removing rows:**

```python
# Instead of deleting outliers, cap them at a reasonable range
# Very useful when you can't afford to lose rows
df['col_clip'] = df['col'].clip(lower=df['col'].quantile(0.01),
                                 upper=df['col'].quantile(0.99))
```

**Ratio features to capture relative comparisons:**

```python
df['ratio_a_to_b']   = df['col_a'] / df['col_b'].clip(lower=1)   # avoid division by zero
df['delta_a_minus_b'] = df['col_a'] - df['col_b']
```

**Cyclical encoding for time features** (better than raw month/hour numbers):

```python
# A model doesn't know that month 12 and month 1 are neighbors — cyclical fixes that
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['dow_sin']   = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos']   = np.cos(2 * np.pi * df['day_of_week'] / 7)
```

**GroupBy-based features — relative position within a group:**

```python
# How does this row compare to others in the same group?
group_col = 'category'
df['group_median']  = df.groupby(group_col)['value'].transform('median')
df['group_std']     = df.groupby(group_col)['value'].transform('std').fillna(1.0)
df['vs_group_ratio']  = df['value'] / df['group_median'].clip(lower=1)
df['vs_group_zscore'] = (df['value'] - df['group_median']) / (df['group_std'] + 1.0)
df['rank_in_group']   = df.groupby(group_col)['value'].rank(method='average', pct=True)
```

**Target Encoding — replace a categorical with the mean of the target per category:**

```python
# SIMPLE version (use when you don't need to worry about leakage)
mean_per_cat = df.groupby('category_col')['target'].mean()
df['cat_encoded'] = df['category_col'].map(mean_per_cat)

# SMOOTHED version (avoids overfit on rare categories)
global_mean = df['target'].mean()
alpha = 10   # smoothing strength
stats = df.groupby('category_col')['target'].agg(['sum', 'count'])
stats['smoothed'] = (stats['sum'] + global_mean * alpha) / (stats['count'] + alpha)
df['cat_encoded'] = df['category_col'].map(stats['smoothed']).fillna(global_mean)
```

> ⚠️ WARNING: For proper target encoding, use K-fold to avoid leakage (see Phase 6)

---

## 1.5 VISUALIZATION

### Check Skewness (important for regression!)

```python
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

for col in numerical_cols:
    skew = df[col].skew()
    print(f"{col}: {skew:.4f}")
    if abs(skew) > 1:
        print(f"  ⚠️  Highly skewed — consider log transform")
```

### Distribution Plots

```python
# Histogram for target variable (ALWAYS plot this first)
plt.figure(figsize=(8, 5))
df['target_col'].hist(bins=40)
plt.title('Target Variable Distribution')
plt.xlabel('Target')
plt.ylabel('Count')
plt.show()
```

### Correlation Heatmap

```python
corr = df.select_dtypes(include=np.number).corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
```

### Scatter Plot (feature vs target)

```python
plt.scatter(df['feature'], df['target'])
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Feature vs Target')
plt.show()

# OR with seaborn:
sns.scatterplot(x='feature', y='target', data=df)
```

### GroupBy Aggregation (for analysis insight)

```python
# Example: average target per category
agg = df.groupby('category_col').agg({'target': ['mean', 'median', 'count']})
print(agg)

# Or with bar plot
df.groupby('category_col')['target'].mean().sort_values().plot(kind='barh')
plt.title('Average Target by Category')
plt.show()
```

---

# PHASE 2: PREPROCESSING

## 2.1 ENCODE CATEGORICAL VARIABLES

```python
# Option A: Label Encoding (for binary or ordinal)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])              # Male=1, Female=0

# Option B: Binary by hand (cleaner for 2 categories)
df['educated'] = (df['education'] == 'Graduate').astype(int)

# Option C: One-Hot Encoding (for nominal with 3+ categories)
df = pd.get_dummies(df, columns=['city'], drop_first=True)
```

## 2.2 TRAIN-TEST SPLIT

```python
X = df.drop('target_col', axis=1)
y = df['target_col']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 80-20 split (or 0.3 for 70-30)
    random_state=42      # reproducibility
    # NOTE: NO stratify for regression (stratify is for classification)
)

print(f"Train size: {X_train.shape}")
print(f"Test size:  {X_test.shape}")
```

> ⚠️ IMPORTANT: No `stratify=y` for regression — stratify only works for classification!

## 2.3 SCALING

```python
scaler = StandardScaler()

# CRITICAL ORDER: fit on train ONLY, then transform both
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
X_test_scaled  = scaler.transform(X_test)         # transform ONLY (no fit!)
```

### 🔑 RobustScaler — Better When Data Has Outliers (from Lab 4)

```python
from sklearn.preprocessing import RobustScaler

# RobustScaler uses median and IQR instead of mean and std
# → Outliers don't distort the scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

**When to use which scaler:**

| Scaler           | How it works                   | Use when                                        |
| ---------------- | ------------------------------ | ----------------------------------------------- |
| `StandardScaler` | Subtract mean, divide by std   | Data is roughly normal, few outliers            |
| `RobustScaler`   | Subtract median, divide by IQR | Data has significant outliers (prices, incomes) |
| `MinMaxScaler`   | Scale to 0–1 range             | You need a bounded range (e.g. neural nets)     |

**Which models NEED scaling?**

| Model                                  | Needs Scaling?           |
| -------------------------------------- | ------------------------ |
| Linear Regression                      | ✅ Yes                   |
| Ridge / Lasso                          | ✅ Yes                   |
| KNN Regressor                          | ✅ Yes (distance-based!) |
| SVR                                    | ✅ Yes                   |
| Decision Tree                          | ❌ No                    |
| Random Forest                          | ❌ No                    |
| Gradient Boosting / XGBoost / LightGBM | ❌ No                    |

> 💡 TIP from prof's notebook: For consistency, you can scale everything and use scaled data for all models. Trees won't be hurt by it.

---

# PHASE 3: REGRESSION MODELS

> From the professor's notebook — these are the exact models he used!

## 3.1 LINEAR REGRESSION

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("Linear Regression R²:  ", r2_score(y_test, y_pred_lr))
```

**When to use:** Baseline model. Good when relationships are roughly linear.

---

## 3.2 RIDGE REGRESSION

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)   # alpha = regularization strength
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

print("Ridge RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_ridge)))
print("Ridge R²:  ", r2_score(y_test, y_pred_ridge))
```

**What is Ridge?** Linear Regression + penalty on large coefficients (L2 regularization).
**When to use:** When features are correlated (multicollinearity) or when Linear Regression overfits.
**alpha:** Higher = stronger penalty = simpler model. Lower = closer to plain Linear Regression.

---

## 3.3 LASSO REGRESSION _(bonus — good to know)_

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)
```

**Difference from Ridge:** Lasso can set some coefficients to exactly 0 → automatic feature selection.

---

## 3.4 KNN REGRESSOR

```python
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=7)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

print("KNN RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_knn)))
print("KNN R²:  ", r2_score(y_test, y_pred_knn))
```

**How it works:** Predicts by averaging the target values of the K nearest training points.
**Must scale:** Yes — KNN is distance-based, unscaled features will dominate.
**n_neighbors:** Try 5, 7, 10. Smaller = more complex model, larger = smoother.

---

## 3.5 DECISION TREE REGRESSOR

```python
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=42, max_depth=5)
dt.fit(X_train, y_train)    # Note: Prof used UNSCALED data here
y_pred_dt = dt.predict(X_test)

print("Decision Tree RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_dt)))
print("Decision Tree R²:  ", r2_score(y_test, y_pred_dt))
```

**max_depth:** Limits tree depth → prevents overfitting. Without it, tree will overfit.
**No scaling needed:** Trees split on thresholds, not distances.

---

## 3.6 SVR (Support Vector Regressor)

```python
from sklearn.svm import SVR

svr = SVR(kernel='rbf', C=10.0, epsilon=0.1)
svr.fit(X_train_scaled, y_train)    # MUST scale
y_pred_svr = svr.predict(X_test_scaled)

print("SVR RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_svr)))
```

**Warning from prof:** SVR is very slow on large datasets. Use a subset if needed:

```python
X_train_sub = X_train_scaled[:2000]   # use first 2000 rows
y_train_sub = y_train[:2000]
svr.fit(X_train_sub, y_train_sub)
```

**Parameters:**

- `kernel='rbf'` — standard choice for non-linear data
- `C` — regularization (higher C = fits training data more tightly)
- `epsilon` — tolerance margin (predictions within epsilon are not penalized)

---

## 3.7 RANDOM FOREST REGRESSOR _(great all-around model)_

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("Random Forest R²:  ", r2_score(y_test, y_pred_rf))
```

---

# PHASE 4: REGRESSION METRICS (NEW — Very Important!)

> This is the big difference from prelims. You must understand all three, plus RMSLE from lab4.

## 4.1 THE FOUR KEY METRICS

### MAE — Mean Absolute Error

```python
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.4f}")
```

- Average of |actual − predicted|
- **Easy to interpret** — same unit as your target
- Less sensitive to outliers
- "On average, my predictions are off by MAE units"

### MSE — Mean Squared Error

```python
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.4f}")
```

- Average of (actual − predicted)²
- **Penalizes large errors heavily**
- Harder to interpret (units are squared)

### RMSE — Root Mean Squared Error ← Prof uses this most!

```python
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")
```

- Square root of MSE → same units as target
- **Most commonly used regression metric**
- Lower is better
- Still penalizes large errors more than MAE

### R² Score — Coefficient of Determination

```python
r2 = r2_score(y_test, y_pred)
print(f"R²: {r2:.4f}")
```

- How much variance in the target the model explains
- **1.0 = perfect**, **0.0 = model is useless**, **negative = worse than just predicting the mean**
- Higher is better

### 🔑 RMSLE — Root Mean Squared LOG Error (from Lab 4!)

```python
# Use when your target spans many orders of magnitude (prices, counts, etc.)
# Kaggle competitions with price targets often use this metric

def rmsle(y_true, y_pred):
    # Clip predictions to avoid log of negative
    y_pred_clipped = np.clip(y_pred, 0, None)
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred_clipped)))

print(f"RMSLE: {rmsle(y_test, y_pred):.4f}")
```

**Why RMSLE instead of RMSE?**

- A $1,000 error on a $10,000 job (10% off) should be penalized the same as a $100,000 error on a $1,000,000 job (also 10% off)
- RMSE would heavily penalize the large job even though the relative error is the same
- RMSLE treats **relative/percentage errors** symmetrically → perfect for prices and counts
- **Lower is better. Typical good RMSLE for price data: < 0.3**

**When to use log1p on the target itself:**

```python
# If your target is heavily right-skewed (prices, salaries, counts)
# Transform BEFORE training, then reverse AFTER predicting

y_train_log = np.log1p(y_train)    # transform
model.fit(X_train, y_train_log)

y_pred_log  = model.predict(X_test)
y_pred_orig = np.expm1(y_pred_log)  # reverse: expm1(x) = e^x - 1

# Then evaluate on the original scale
rmse = np.sqrt(mean_squared_error(y_test, y_pred_orig))
```

## 4.2 METRIC SUMMARY TABLE

| Metric | Good Value         | Unit           | Use When                       |
| ------ | ------------------ | -------------- | ------------------------------ |
| MAE    | As low as possible | Same as target | Robust to outliers needed      |
| RMSE   | As low as possible | Same as target | Standard regression            |
| R²     | Close to 1.0       | Unitless (0–1) | Explaining variance            |
| RMSLE  | As low as possible | Unitless       | Skewed targets, prices, counts |

> **In the exam:** Always print ALL THREE standard metrics (RMSE, MAE, R²). If target is skewed (prices, counts), add RMSLE. Then in your insight, say which model has lowest RMSE and highest R².

---

# PHASE 5: MODEL COMPARISON (Regression Version)

```python
# Store all predictions in a dict
models_preds = {
    'Linear Regression': y_pred_lr,
    'Ridge':             y_pred_ridge,
    'KNN':               y_pred_knn,
    'Decision Tree':     y_pred_dt,
    'SVR':               y_pred_svr,
    # 'Random Forest':   y_pred_rf,
}

# Build comparison table
results = []
for name, preds in models_preds.items():
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    results.append({'Model': name, 'RMSE': round(rmse, 4),
                    'MAE': round(mae, 4), 'R²': round(r2, 4)})

results_df = pd.DataFrame(results).sort_values('RMSE')   # sort by RMSE ascending
print(results_df.to_string(index=False))
```

### Visualize Comparison

```python
# RMSE bar chart (lower = better)
plt.figure(figsize=(10, 5))
sns.barplot(x='RMSE', y='Model', data=results_df, palette='magma')
plt.title('Model Comparison: RMSE (Lower is Better)')
plt.xlabel('RMSE')
plt.tight_layout()
plt.show()

# R² bar chart (higher = better)
plt.figure(figsize=(10, 5))
sns.barplot(x='R²', y='Model', data=results_df.sort_values('R²', ascending=False), palette='viridis')
plt.title('Model Comparison: R² Score (Higher is Better)')
plt.xlabel('R² Score')
plt.tight_layout()
plt.show()
```

### Predicted vs Actual Plot

```python
# Great visualization for regression — shows how well predictions match reality
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2)   # perfect prediction line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted — Linear Regression')
plt.tight_layout()
plt.show()
```

> 💡 Points closer to the red dashed line = better predictions

---

# PHASE 6: HYPERPARAMETER TUNING (NEW!)

> This is the completely new topic for midterm. From the Tuning notebook.

## What are Hyperparameters?

- Settings you configure **before** training (not learned from data)
- Examples: `max_depth`, `n_estimators`, `alpha`, `n_neighbors`, `C`
- Tuning = finding the best combo of these settings

## 6.1 CROSS-VALIDATION (Understand This First!)

```python
from sklearn.model_selection import cross_val_score

model = RandomForestRegressor(random_state=42)

# cv=5 means 5-fold: splits data into 5 parts, trains 5 times
cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                             scoring='neg_mean_squared_error')

# Convert to RMSE (scores are negative by convention)
rmse_scores = np.sqrt(-cv_scores)
print(f"CV RMSE scores: {rmse_scores}")
print(f"Mean CV RMSE:   {rmse_scores.mean():.4f}")
print(f"Std CV RMSE:    {rmse_scores.std():.4f}")
```

**Why use cross-validation?**

- Prevents overfitting to a specific train-test split
- Gives more reliable estimate of true model performance
- The `± std` tells you how stable/consistent the model is

**For classification (like in the prof's notebook):**

```python
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
print(f"Average CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
```

---

## 6.2 GRIDSEARCHCV — Exhaustive Search

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor    # or Classifier

# Step 1: Define parameter grid (all combinations will be tried)
param_grid = {
    'n_estimators':    [50, 100, 200],
    'max_depth':       [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
# Total combinations: 3 × 4 × 3 = 36 combos × 5 folds = 180 fits

# Step 2: Create GridSearchCV
grid_search = GridSearchCV(
    estimator  = RandomForestRegressor(random_state=42),
    param_grid = param_grid,
    cv         = 5,
    scoring    = 'neg_mean_squared_error',   # for regression
    # scoring  = 'accuracy',                 # for classification
    n_jobs     = -1,     # use all CPU cores (faster)
    verbose    = 1       # shows progress
)

# Step 3: Fit (this trains ALL combinations)
grid_search.fit(X_train, y_train)

# Step 4: Get results
print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:  ", np.sqrt(-grid_search.best_score_))  # convert for RMSE

# Step 5: Get best model (already trained with best params)
best_model = grid_search.best_estimator_
```

**Pros:** Tries every combination → guaranteed to find best in grid
**Cons:** Very slow with many parameters

---

## 6.3 RANDOMIZEDSEARCHCV — Faster Alternative

```python
from sklearn.model_selection import RandomizedSearchCV

# Can use ranges (np.arange) instead of exact lists
param_dist = {
    'n_estimators':      np.arange(50, 500, 50),    # 50,100,150,...450
    'max_depth':         [None, 10, 20, 30, 40, 50],
    'min_samples_split': np.arange(2, 15, 2),
    'bootstrap':         [True, False]
}

random_search = RandomizedSearchCV(
    estimator            = RandomForestRegressor(random_state=42),
    param_distributions  = param_dist,
    n_iter               = 20,      # only try 20 random combos (not all!)
    cv                   = 3,
    scoring              = 'neg_mean_squared_error',
    n_jobs               = -1,
    random_state         = 42,
    verbose              = 1
)

random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
best_random_model = random_search.best_estimator_
```

**Pros:** Much faster — only tries `n_iter` combinations
**Cons:** Not guaranteed to find the absolute best, but usually finds a very good one

---

## 6.4 COMPARE BASELINE vs TUNED MODELS

```python
# Get predictions from all 3 versions
y_pred_baseline = baseline_model.predict(X_test)
y_pred_grid     = best_model.predict(X_test)
y_pred_random   = best_random_model.predict(X_test)

# Build comparison table
comparison = pd.DataFrame({
    'Model':    ['Baseline', 'Grid Search Tuned', 'Random Search Tuned'],
    'RMSE':     [
        np.sqrt(mean_squared_error(y_test, y_pred_baseline)),
        np.sqrt(mean_squared_error(y_test, y_pred_grid)),
        np.sqrt(mean_squared_error(y_test, y_pred_random))
    ],
    'R²':       [
        r2_score(y_test, y_pred_baseline),
        r2_score(y_test, y_pred_grid),
        r2_score(y_test, y_pred_random)
    ]
})

print(comparison.to_string(index=False))
```

---

## 6.5 GRIDSEARCHCV FOR OTHER MODELS

### Ridge / Lasso Tuning

```python
param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train_scaled, y_train)
print("Best alpha:", grid.best_params_)
```

### KNN Tuning

```python
param_grid = {'n_neighbors': [3, 5, 7, 10, 15, 20]}
grid = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5,
                    scoring='neg_mean_squared_error')
grid.fit(X_train_scaled, y_train)
print("Best k:", grid.best_params_)
```

### Decision Tree Tuning

```python
param_grid = {
    'max_depth':        [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}
grid = GridSearchCV(DecisionTreeRegressor(random_state=42),
                    param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
```

---

# PHASE 7: FULL WORKFLOW TEMPLATE

```python
# ──────────────────────────────────────────────────────────
# STEP 1: LOAD & INSPECT
# ──────────────────────────────────────────────────────────
df = pd.read_csv('dataset.csv')
print(df.shape)
df.info()
df.describe()
df.isnull().sum()

# ──────────────────────────────────────────────────────────
# STEP 2: CLEAN
# ──────────────────────────────────────────────────────────
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)                    # or fill: df['col'].fillna(df['col'].median())
df.drop(['id_column'], axis=1, inplace=True)

# ──────────────────────────────────────────────────────────
# STEP 3: FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────
df['new_feature'] = df['col1'] / df['col2']

# ──────────────────────────────────────────────────────────
# STEP 4: VISUALIZE + ANALYZE
# ──────────────────────────────────────────────────────────
# [histogram of target, heatmap, scatter plots, groupby]

# ──────────────────────────────────────────────────────────
# STEP 5: PREPROCESS
# ──────────────────────────────────────────────────────────
# Encode categorical
df['cat_col'] = LabelEncoder().fit_transform(df['cat_col'])

# Split
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ──────────────────────────────────────────────────────────
# STEP 6: TRAIN MODELS
# ──────────────────────────────────────────────────────────
models = {
    'Linear Regression': LinearRegression(),
    'Ridge':             Ridge(alpha=1.0),
    'KNN':               KNeighborsRegressor(n_neighbors=7),
    'Decision Tree':     DecisionTreeRegressor(random_state=42, max_depth=5),
}

results = []
for name, model in models.items():
    # Use unscaled for tree-based; scaled for linear/knn
    use_scaled = name not in ['Decision Tree', 'Random Forest']
    Xt = X_train_scaled if use_scaled else X_train.values
    Xe = X_test_scaled  if use_scaled else X_test.values

    model.fit(Xt, y_train)
    preds = model.predict(Xe)

    results.append({
        'Model': name,
        'RMSE':  round(np.sqrt(mean_squared_error(y_test, preds)), 4),
        'MAE':   round(mean_absolute_error(y_test, preds), 4),
        'R²':    round(r2_score(y_test, preds), 4)
    })

results_df = pd.DataFrame(results).sort_values('RMSE')
print(results_df.to_string(index=False))

# ──────────────────────────────────────────────────────────
# STEP 7: TUNE BEST MODEL
# ──────────────────────────────────────────────────────────
param_grid = {
    'max_depth':         [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}
grid = GridSearchCV(DecisionTreeRegressor(random_state=42),
                    param_grid, cv=5,
                    scoring='neg_mean_squared_error', n_jobs=-1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)

best_model = grid.best_estimator_
y_pred_tuned = best_model.predict(X_test)
print("Tuned RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_tuned)))
print("Tuned R²:  ", r2_score(y_test, y_pred_tuned))
```

---

# PHASE 7: ADVANCED MODELS — GRADIENT BOOSTING (From Lab 4)

> These are the most powerful regression models used in Kaggle competitions. Worth knowing even if the exam uses simpler models — they show depth.

## 7.1 WHAT IS GRADIENT BOOSTING?

All three models below (XGBoost, LightGBM, CatBoost) are **Gradient Boosting** algorithms:

- Build trees **sequentially** — each tree corrects the errors of the previous one
- Very powerful out of the box; even better after tuning
- **Don't need scaling** (tree-based)
- Support **early stopping** — stop training when validation score stops improving

**Analogy:** Like a team where each new member specifically targets the mistakes the previous members made.

**vs Random Forest:**

- Random Forest: trees are built **in parallel** (independently), averaged
- Gradient Boosting: trees are built **sequentially**, each improving on the last
- Gradient Boosting is usually more accurate but slower to train

---

## 7.2 XGBoost

```python
import xgboost as xgb

# Basic usage
xgb_model = xgb.XGBRegressor(
    n_estimators    = 500,        # number of trees
    learning_rate   = 0.05,       # step size per tree (lower = more trees needed)
    max_depth       = 5,          # max depth of each tree
    subsample       = 0.8,        # fraction of rows used per tree
    colsample_bytree= 0.8,        # fraction of features used per tree
    reg_alpha       = 0.5,        # L1 regularization
    reg_lambda      = 5.0,        # L2 regularization
    random_state    = 42,
    n_jobs          = -1
)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# With early stopping (stops when val score doesn't improve for 50 rounds)
xgb_model = xgb.XGBRegressor(
    n_estimators=2000, learning_rate=0.05,
    early_stopping_rounds=50,
    random_state=42
)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
print("Best iteration:", xgb_model.best_iteration)
```

**Key parameters to tune:**
| Param | What it does | Typical range |
|---|---|---|
| `n_estimators` | Number of trees | 100–3000 |
| `learning_rate` | Step size | 0.01–0.1 |
| `max_depth` | Tree depth | 3–8 |
| `subsample` | Row sampling | 0.6–1.0 |
| `colsample_bytree` | Feature sampling | 0.5–1.0 |
| `reg_alpha` | L1 penalty | 0.01–10 |
| `reg_lambda` | L2 penalty | 1–20 |

---

## 7.3 LightGBM

```python
from lightgbm import LGBMRegressor
import lightgbm as lgb

lgb_model = LGBMRegressor(
    n_estimators    = 500,
    learning_rate   = 0.05,
    num_leaves      = 64,        # key LGB param — controls complexity (not max_depth)
    min_child_samples=40,         # min samples in a leaf — prevents overfit
    subsample       = 0.8,
    colsample_bytree= 0.8,
    reg_alpha       = 0.2,
    reg_lambda      = 4.0,
    random_state    = 42,
    n_jobs          = -1,
    verbosity       = -1          # suppress output
)

# With early stopping (LightGBM syntax)
lgb_model = LGBMRegressor(n_estimators=2000, learning_rate=0.05, verbosity=-1)
lgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
print("Best iteration:", lgb_model.best_iteration_)

y_pred = lgb_model.predict(X_test)
```

**LightGBM vs XGBoost:**

- LightGBM grows trees **leaf-wise** (deeper, more asymmetric)
- XGBoost grows trees **level-wise** (balanced)
- LightGBM is usually **faster** and handles large datasets better
- Use `num_leaves` to control complexity in LGB (not `max_depth`)

---

## 7.4 FEATURE IMPORTANCE — Tree Models

```python
# After training XGBoost or LightGBM or Random Forest
import pandas as pd

feat_imp = pd.Series(
    xgb_model.feature_importances_,
    index=X_train.columns    # X_train must be DataFrame, not numpy array
)
feat_imp = feat_imp.sort_values(ascending=False).head(20)

plt.figure(figsize=(10, 7))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
plt.title('Feature Importance — XGBoost (Top 20)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()
```

> 💡 Features in the top 10 of ALL your models are your most reliable signals. Mention this in your insight!

---

# PHASE 8: ENSEMBLE METHODS (From Lab 4)

> Using multiple models together almost always beats any single model.

## 8.1 SIMPLE AVERAGING (Weighted Blend)

```python
# Train multiple models first, then average their predictions

y_pred_lr  = lr_model.predict(X_test)
y_pred_rf  = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

# Equal blend
y_pred_blend = (y_pred_lr + y_pred_rf + y_pred_xgb) / 3

# Weighted blend (give more weight to better models)
y_pred_blend = 0.2 * y_pred_lr + 0.3 * y_pred_rf + 0.5 * y_pred_xgb

rmse_blend = np.sqrt(mean_squared_error(y_test, y_pred_blend))
print(f"Blended RMSE: {rmse_blend:.4f}")
```

**Why blending works:** Different models make different kinds of errors. When you average them, those errors cancel each other out partially → better overall.

---

## 8.2 STACKING (Meta-Learning)

```python
# Step 1: Train base models using cross-validation, collect OOF (out-of-fold) predictions
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_rf  = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))

for tr_i, va_i in kf.split(X_train):
    Xtr, Xva = X_train.iloc[tr_i], X_train.iloc[va_i]
    ytr       = y_train.iloc[tr_i]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(Xtr, ytr)
    oof_rf[va_i] = rf.predict(Xva)

    xgb_m = xgb.XGBRegressor(n_estimators=200, random_state=42)
    xgb_m.fit(Xtr, ytr)
    oof_xgb[va_i] = xgb_m.predict(Xva)

# Step 2: Stack OOF predictions as new features
meta_X_train = np.column_stack([oof_rf, oof_xgb])

# Get test predictions from models trained on ALL training data
rf_full  = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
xgb_full = xgb.XGBRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)
meta_X_test = np.column_stack([rf_full.predict(X_test),
                                xgb_full.predict(X_test)])

# Step 3: Train meta-model (usually Ridge or Linear Regression) on OOF predictions
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_X_train, y_train)
y_pred_stacked = meta_model.predict(meta_X_test)

print("Stacked RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_stacked)))
```

**Why Ridge for the meta-model?** Only a few meta-features → linear is enough. Ridge prevents any single model from dominating.

---

## 8.3 GroupKFold — When Your Data Has Groups (From Lab 4)

```python
from sklearn.model_selection import GroupKFold

# Use when rows belonging to the same entity should stay together in the same fold
# Example: multiple bids per job_id — you don't want the same job in both train and val

gkf = GroupKFold(n_splits=5)
groups = df['job_id']   # the grouping column

for tr_i, va_i in gkf.split(X, y, groups=groups):
    X_tr, X_va = X.iloc[tr_i], X.iloc[va_i]
    y_tr, y_va = y.iloc[tr_i], y.iloc[va_i]
    # ... train and evaluate
```

**Regular KFold vs GroupKFold:**
| | Regular KFold | GroupKFold |
|---|---|---|
| Use when | Rows are independent | Rows share a group (same customer, same job) |
| Problem with regular KFold | Same entity in train AND val → overoptimistic score | N/A |
| Real-world examples | Survey responses | Multiple bids, repeat customers |

---

# PHASE 9: ADVANCED HYPERPARAMETER TUNING (From Lab 4)

## 9.1 OPTUNA — Bayesian Optimization

> GridSearch and RandomSearch try hyperparameters blindly. Optuna _learns_ from previous trials and focuses on promising regions. Much more efficient for large parameter spaces.

```python
# pip install optuna
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    # Step 1: Define parameters to search
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate':    trial.suggest_float('lr', 0.01, 0.1, log=True),
        'max_depth':        trial.suggest_int('max_depth', 3, 8),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('col_bytree', 0.5, 1.0),
    }

    # Step 2: Train and evaluate with cross-validation
    model = xgb.XGBRegressor(random_state=42, **params)
    scores = cross_val_score(model, X_train, y_train,
                              cv=3, scoring='neg_mean_squared_error')
    return float(np.sqrt(-scores.mean()))   # return RMSE (to minimize)

# Step 3: Run the study
study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=42)   # TPE = Bayesian method
)
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("Best params:", study.best_params)
print("Best RMSE:  ", study.best_value)

# Step 4: Use best params
best_model = xgb.XGBRegressor(random_state=42, **study.best_params)
best_model.fit(X_train, y_train)
```

**Optuna vs GridSearch vs RandomSearch:**

| Method               | How it searches             | Speed   | Best for                         |
| -------------------- | --------------------------- | ------- | -------------------------------- |
| `GridSearchCV`       | All combinations            | Slowest | Small param grids (< 100 combos) |
| `RandomizedSearchCV` | Random samples              | Fast    | Medium param spaces              |
| `Optuna`             | Bayesian (learns from past) | Fastest | Large param spaces (7+ params)   |

**Suggest functions:**

```python
trial.suggest_int('param', low, high)              # integer
trial.suggest_float('param', low, high)            # float
trial.suggest_float('param', low, high, log=True)  # log scale (for lr, alpha)
trial.suggest_categorical('param', ['a','b','c'])  # categorical
```

---

## 9.2 EARLY STOPPING AS IMPLICIT TUNING

```python
# Instead of manually setting n_estimators, let early stopping find the best one

xgb_model = xgb.XGBRegressor(
    n_estimators=3000,          # set high — early stopping will cut it short
    learning_rate=0.02,
    early_stopping_rounds=50,   # stop if no improvement for 50 rounds
    random_state=42
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],   # monitor validation score
    verbose=False
)

best_n = xgb_model.best_iteration
print(f"Best number of trees: {best_n}")

# For final prediction, retrain with best_n (no early stopping)
final_model = xgb.XGBRegressor(n_estimators=best_n, learning_rate=0.02, random_state=42)
final_model.fit(X_train, y_train)
```

---

# PHASE 10: SAMPLE WEIGHTS (From Lab 4)

> When some training rows are more important/relevant than others.

```python
# Use case: You want recent data to matter more than old data
# (e.g. 2024 patterns are more relevant to 2025 predictions than 2018 patterns)

year_weight_map = {2024: 4.0, 2023: 2.0, 2022: 1.0, 2021: 0.5, 2020: 0.25}
sample_weights = df['year'].map(year_weight_map).fillna(0.1).values
# Normalize so mean weight = 1
sample_weights = sample_weights / sample_weights.mean()

# Pass to model's fit method
model.fit(X_train, y_train, sample_weight=sample_weights)

# Works with: Ridge, XGBoost, LightGBM, RandomForest, GradientBoosting
```

---

## What is Regularization? (Ridge / Lasso)

Regular Linear Regression just minimizes error. Ridge/Lasso also penalize complexity:

- **Without regularization:** Model might memorize training data (overfit) → bad on test data
- **With regularization (alpha > 0):** Model is forced to keep coefficients small → generalizes better
- `alpha` knob: higher = more penalty = simpler model
- **Ridge (L2):** Shrinks all coefficients toward zero, keeps them all
- **Lasso (L1):** Can set some coefficients to exactly zero → automatic feature selection

## What is Overfitting vs Underfitting?

- **Overfitting:** Great on train, bad on test → train RMSE low, test RMSE high
  - Fix: regularization, limit `max_depth`, cross-validation, reduce features
- **Underfitting:** Bad on both → high RMSE everywhere
  - Fix: more complex model, more features, less regularization

## What is Cross-Validation (CV)?

- CV splits training data into `k` folds, trains and tests `k` times → average score
- More reliable than single train/test split
- `cv=5` = 5-fold (most common). Lower std across folds = more stable model.

## Why Log-Transform a Skewed Target?

- Prices, salaries, counts are right-skewed → a few huge values pull the distribution
- Linear models assume normally distributed errors — skewed target violates this
- Log-transform makes distribution more Gaussian → better model fit
- Always use `np.log1p()` (safe for zeros), reverse with `np.expm1()` after predicting

## GridSearch vs RandomSearch vs Optuna?

|          | GridSearchCV     | RandomizedSearchCV     | Optuna                   |
| -------- | ---------------- | ---------------------- | ------------------------ |
| Tries    | ALL combinations | Random `n_iter` combos | Smart Bayesian combos    |
| Speed    | Slowest          | Fast                   | Fastest for large spaces |
| Best for | Small grids      | Medium spaces          | 5+ parameters            |

## What is Stacking / Ensembling?

- **Blending:** Weighted average of predictions from different models
- **Stacking:** Use model predictions as inputs to a "meta-model" (usually Ridge)
- Works because different models make different mistakes → averaging cancels errors
- **OOF (out-of-fold) predictions** must be used to train meta-model to avoid leakage

## What is Target Encoding?

- Replace a categorical column with the average target value per category
- More informative than one-hot encoding for high-cardinality categoricals
- Risk: **leakage** if you use target from the same rows → always use smoothing or K-fold TE

---

# ⚡ INSIGHTS TO WRITE (Copy & Adapt)

### After Data Inspection:

> "The dataset contains X rows and Y columns. The target variable is [name], which is a continuous numeric variable indicating [what it means]. This is a **regression problem**. Features include a mix of numeric and categorical variables."

### After Cleaning:

> "X duplicate rows were removed. Y missing values were found in column Z and were filled using the median, as the distribution appeared right-skewed. Outliers in column W were removed using the IQR method, reducing the dataset by X rows."

### After Target Distribution Plot:

> "The target variable [name] is heavily right-skewed (skewness = X.XX), with values spanning a wide range from [min] to [max]. A log transformation was applied to normalize the distribution before modeling, which reduced skewness to X.XX."

### After Feature Engineering:

> "A new feature [name] was created by [rationale]. This captures [insight] which may help the model better predict [target]."

### After Correlation Heatmap:

> "The heatmap reveals that [feature A] has the strongest positive correlation with [target] (r = X.XX). [Feature B] and [Feature C] are highly correlated with each other (r > 0.9), suggesting possible multicollinearity. For Ridge regression, this is handled by the regularization penalty."

### After Model Training:

> "Among the trained models, [Model Name] achieved the lowest RMSE of X.XX and the highest R² of X.XX, indicating it explains XX% of the variance in the target variable. Decision Tree and Linear Regression underperformed, likely due to [reason]."

### After Tuning:

> "After applying GridSearchCV with parameters [list], the best configuration was [params]. The tuned model improved RMSE from X.XX to Y.YY, demonstrating the value of hyperparameter optimization."

### After Cross-Validation:

> "5-fold cross-validation yielded an average RMSE of X.XX ± Y.YY, confirming that the model generalizes consistently and is not overfitting to a particular split."

### After RMSLE Metric (if target is skewed/price data):

> "RMSLE was used as the primary metric because the target spans several orders of magnitude. RMSLE treats relative errors equally regardless of scale — a 10% error on a small bid is penalized the same as a 10% error on a large bid."

### After Feature Importance:

> "[Feature A] and [Feature B] are the most important predictors across all three models, suggesting they capture the dominant patterns in the data. Features appearing in the top 10 of all models are the most reliable signals."

### After Ensembling:

> "A blended ensemble (X% model A + Y% model B) achieved RMSE of X.XX, outperforming all individual models. This improvement demonstrates that model diversity reduces prediction variance — each model's errors partially cancel out."

---

# 🗓️ EXAM DAY CHECKLIST

### STEP 1: Load & Inspect (5 min)

- [ ] `df.head()`, `df.info()`, `df.describe()`, `df.shape`
- [ ] Identify target variable (continuous = regression!)
- [ ] Write insight: dataset overview

### STEP 2: Clean (10 min)

- [ ] `df.drop_duplicates()`
- [ ] `df.isnull().sum()` → fill or drop
- [ ] Check and handle outliers (IQR)
- [ ] Fix data types if needed
- [ ] Write insight: what you cleaned and why

### STEP 3: Feature Engineering (5–10 min)

- [ ] Drop ID / irrelevant columns
- [ ] Create at least 1 new meaningful feature
- [ ] Extract from dates if applicable
- [ ] Write insight: why these features help

### STEP 4: Visualize (10–15 min)

- [ ] Target distribution histogram
- [ ] Correlation heatmap
- [ ] Scatter plot: key feature vs target
- [ ] GroupBy aggregation bar chart
- [ ] **Write insight for EACH visualization**

### STEP 5: Preprocess (5–10 min)

- [ ] Encode categorical columns
- [ ] `train_test_split` (no stratify for regression)
- [ ] `StandardScaler` — fit on train, transform test
- [ ] Write insight: preprocessing choices

### STEP 6: Train 3+ Models (10–15 min)

- [ ] At minimum: Linear Regression + Ridge + KNN (or Decision Tree)
- [ ] Print RMSE and R² for each
- [ ] Write insight

### STEP 7: Tune (10 min)

- [ ] `cross_val_score` on your best model
- [ ] `GridSearchCV` or `RandomizedSearchCV`
- [ ] Compare baseline vs tuned
- [ ] Write insight: did tuning help?

### STEP 8: Compare (5 min)

- [ ] `results_df` table sorted by RMSE
- [ ] Bar chart for visual comparison
- [ ] Final insight: which model won and why

---

# 🔥 COMMON MISTAKES TO AVOID

1. **Using `stratify=y` for regression** → Error! Remove it.
2. **Using accuracy as metric** → That's for classification. Use RMSE and R² for regression.
3. **Fitting scaler on test data** → Data leakage! `fit_transform` only on train.
4. **Comparing RMSE without checking units** → Always mention what the RMSE means in context.
5. **Not using `random_state=42`** → Results change every run, non-reproducible.
6. **GridSearch without baseline** → Always train a baseline first to show improvement.
7. **Not printing all metrics** → Lost points! Show RMSE, MAE, R² for every model.
8. **Forgetting insights on visualizations** → Easy points lost.
9. **Using `GridSearchCV.best_score_` directly for regression** → It's negative (neg*MSE), convert with `np.sqrt(-grid.best_score*)`.
10. **Setting `n_iter` too high in RandomizedSearch** → Slow exam; keep it at 10–20.
11. **Log-transforming target but forgetting to reverse** → Predicting in log space, evaluating in original scale → wrong metrics. Always `np.expm1()` your predictions.
12. **Scaling before train-test split** → Data leakage! Split first, then scale.
13. **Using `log()` instead of `log1p()`** → `log(0)` is undefined; `log1p(0) = 0`. Always prefer `log1p`.
14. **Training on all rows when some have no target** → If target is null/zero for some rows, drop them before training.
15. **Blending without checking if models are diverse** → Blending two identical models does nothing. Models should have different structures (tree vs linear, etc.).

---

# 🏆 BONUS — FEATURE IMPORTANCE (for tree-based models)

```python
# After fitting a Random Forest or Decision Tree
feature_names = X_train.columns  # if X_train is a DataFrame
importances   = rf.feature_importances_

feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_df = feat_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_df)
plt.title('Feature Importance — Random Forest')
plt.tight_layout()
plt.show()
```

---

## END OF REVIEWER

**Good luck on your midterm! 🚀**

Remember:

- Regression = predicting NUMBERS → use RMSE, MAE, R² (+ RMSLE for skewed/price targets)
- Always train a baseline → then tune with GridSearch or RandomSearch
- Cross-validation gives you confidence your model isn't just lucky
- Log-transform skewed targets → reverse with expm1 after predicting
- XGBoost/LightGBM outperform basic models on most real datasets
- Ensembling (blend or stack) almost always beats any single model
- Insights at every step = easy points!

---

_Reviewer covers: Prof's Regression Demo + Tuning Notebook + Lab 4 Kaggle Competition (v1.9)_
