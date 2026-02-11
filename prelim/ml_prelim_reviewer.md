# 🎯 MACHINE LEARNING PRELIM EXAM REVIEWER

## **EXAM STRUCTURE OVERVIEW**

1. Exploratory Data Analysis (EDA)
2. Data Preprocessing
3. Model Training (3+ algorithms)
4. Model Comparison & Evaluation

---

# PHASE 1: EXPLORATORY DATA ANALYSIS (EDA)

## 📊 **1.1 DATA INSPECTION**

### Sub-steps:

1. **Load the dataset**
2. **Check basic information**
3. **Examine first/last rows**
4. **Check data types**
5. **Check dataset shape**
6. **Identify target variable**

### What to Look For:

- Number of rows and columns
- Column names and their meaning
- Data types (numeric vs categorical vs datetime)
- Which column is your target/label
- Are there any obvious issues?

### Common Pandas Functions:

```python
import pandas as pd
import numpy as np

# Loading data
df = pd.read_csv('data.csv')

# Basic inspection
df.head()                    # First 5 rows
df.tail()                    # Last 5 rows
df.info()                    # Data types, non-null counts, memory
df.describe()                # Statistical summary (mean, std, min, max, quartiles)
df.shape                     # (rows, columns)
df.columns                   # Column names
df.dtypes                    # Data type of each column
df.nunique()                 # Number of unique values per column
df.value_counts('column')    # Count occurrences of each value
```

### WHY You Do It:

- **Understand your data before making assumptions**
- Identify the problem type (classification vs regression)
- Spot potential issues early (missing data, wrong types)
- Know what features you're working with

### Tips:

- **If you see:** Many columns → Note which are features vs target
- **If you see:** Mixed data types → Separate numeric and categorical
- **If you see:** Date columns → Might need feature engineering
- Write down: "The dataset has X rows and Y columns. The target variable is Z for [classification/regression]."

---

## 🔍 **1.2 DATA CLEANING**

### Sub-steps:

1. **Check for missing values**
2. **Check for duplicates**
3. **Identify and handle outliers**
4. **Fix data types if needed**
5. **Handle text data (if any)**

### What to Look For:

- How many missing values per column?
- Are there duplicate rows?
- Are there extreme outliers?
- Are numeric columns stored as strings?
- Do text columns need cleaning?

### Common Pandas Functions:

```python
# Missing Values
df.isnull().sum()            # Count nulls per column
df.isnull().sum() / len(df) * 100  # Percentage of nulls
df.isna().any()              # Which columns have nulls

# Duplicates
df.duplicated().sum()        # Count duplicate rows
df.drop_duplicates()         # Remove duplicates

# Outliers Detection (IQR method)
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['column'] < lower_bound) | (df['column'] > upper_bound)]

# Data type conversion
df['column'] = pd.to_numeric(df['column'], errors='coerce')
df['date'] = pd.to_datetime(df['date'])

# Text cleaning
df['text'] = df['text'].str.lower()           # Lowercase
df['text'] = df['text'].str.strip()           # Remove whitespace
df['text'] = df['text'].str.replace('[^\w\s]', '')  # Remove special chars
```

### HANDLING MISSING VALUES - Decision Tree:

**WHEN TO DROP:**

- < 5% missing → Usually safe to drop rows
- Column has > 70% missing → Drop entire column
- Missing values are random and sample size is large

**WHEN TO FILL (IMPUTE):**

- Numeric columns:
  - **Mean**: Data is normally distributed, no outliers
  - **Median**: Data has outliers (more robust)
  - **Mode**: Categorical-like numeric (e.g., number of rooms: 1,2,3)
  - **Forward/Backward Fill**: Time series data
- Categorical columns:
  - **Mode**: Most common category
  - **"Unknown"**: New category for missing
- **Advanced**: Use model-based imputation (KNN, iterative)

```python
# Dropping
df = df.dropna()                    # Drop all rows with any NaN
df = df.dropna(subset=['column'])   # Drop rows where 'column' is NaN
df = df.drop('column', axis=1)      # Drop entire column

# Filling
df['numeric_col'].fillna(df['numeric_col'].mean(), inplace=True)
df['numeric_col'].fillna(df['numeric_col'].median(), inplace=True)
df['category_col'].fillna(df['category_col'].mode()[0], inplace=True)
df['category_col'].fillna('Unknown', inplace=True)

# Advanced imputation
from sklearn.impute import SimpleImputer, KNNImputer

imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

knn_imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])
```

### WHY You Do It:

- **Missing values** break most ML algorithms → Need to handle them
- **Duplicates** can bias your model and waste computation
- **Outliers** can skew models (especially linear ones)
- **Wrong data types** prevent proper analysis
- **Clean data = Better model performance**

### Tips:

- **If you see:** > 30% missing in a column → Consider dropping column
- **If you see:** Only a few missing values → Drop those rows
- **If you see:** Outliers in income/price → Use median, not mean
- **If you see:** Text with inconsistent case → Normalize to lowercase
- Always document what you cleaned and why!

---

## 🏗️ **1.3 FEATURE ENGINEERING**

### Sub-steps:

1. **Create new features from existing ones**
2. **Extract features from dates**
3. **Bin/discretize continuous variables**
4. **Combine features**
5. **Drop useless features**

### What to Look For:

- Can you create ratios or differences?
- Are there date columns to extract from?
- Should continuous variables be grouped?
- Are there highly correlated features?
- Are there constant or ID columns?

### Common Pandas Functions:

```python
# Creating new features
df['total_price'] = df['quantity'] * df['unit_price']
df['bmi'] = df['weight'] / (df['height'] ** 2)

# Date feature extraction
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Binning
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100],
                         labels=['Child', 'Young Adult', 'Adult', 'Senior'])
df['income_bracket'] = pd.qcut(df['income'], q=4,
                               labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Dropping useless features
df = df.drop(['id', 'name'], axis=1)  # IDs and names usually don't help

# Check correlation
correlation_matrix = df.corr()
# If two features have correlation > 0.9, consider dropping one
```

### WHY You Do It:

- **Create meaningful features** that help the model learn patterns
- **Extract hidden information** (e.g., month from date affects sales)
- **Reduce complexity** by grouping continuous variables
- **Remove noise** from irrelevant features
- Better features → Better model performance

### Tips:

- **If you see:** DateTime → Extract year, month, day, day_of_week
- **If you see:** Two related numbers → Try ratios or differences
- **If you see:** Age, Income, Price → Consider binning into categories
- **If you see:** ID or Name columns → Drop them (unless it's test data requirement)
- **If you see:** Two features with 0.95+ correlation → Keep one, drop the other

---

## 📈 **1.4 VISUALIZATION & ANALYSIS**

### Sub-steps:

1. **Univariate analysis** (one variable at a time)
2. **Bivariate analysis** (two variables)
3. **Multivariate analysis** (multiple variables)
4. **Distribution analysis**
5. **Relationship analysis**

### What to Look For:

- Distribution of target variable (balanced?)
- Distribution of features
- Relationships between features and target
- Correlations between features
- Patterns or clusters

### Common Plotting Functions:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# UNIVARIATE - Numeric
df['column'].hist(bins=30)
plt.title('Distribution of Column')
plt.show()

sns.boxplot(x=df['column'])  # Shows outliers
plt.title('Boxplot of Column')
plt.show()

# UNIVARIATE - Categorical
df['category'].value_counts().plot(kind='bar')
plt.title('Count of Categories')
plt.show()

sns.countplot(x='category', data=df)
plt.title('Count Plot')
plt.show()

# BIVARIATE - Numeric vs Numeric
plt.scatter(df['feature1'], df['feature2'])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

sns.scatterplot(x='feature1', y='feature2', data=df)

# BIVARIATE - Categorical vs Numeric
sns.boxplot(x='category', y='numeric', data=df)
plt.title('Numeric values across Categories')
plt.show()

sns.violinplot(x='category', y='numeric', data=df)

# BIVARIATE - Categorical vs Categorical
pd.crosstab(df['cat1'], df['cat2']).plot(kind='bar')

# CORRELATION HEATMAP (Multivariate)
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# PAIRPLOT (Multiple features)
sns.pairplot(df, hue='target')
plt.show()
```

### Common Aggregation Functions:

```python
# GroupBy aggregations
df.groupby('category')['value'].mean()
df.groupby('category')['value'].agg(['mean', 'median', 'std', 'count'])
df.groupby(['cat1', 'cat2'])['value'].sum()

# Pivot tables
pd.pivot_table(df, values='sales', index='product',
               columns='month', aggfunc='sum')

# General aggregations
df['column'].mean()
df['column'].median()
df['column'].std()
df['column'].min()
df['column'].max()
df['column'].sum()
df['column'].count()
```

### WHY You Do It:

- **Visualize patterns** that numbers alone can't show
- **Identify relationships** between features and target
- **Detect imbalances** in classification problems
- **Find insights** to report in your exam
- **Professors LOVE seeing insights** → Easy points!

### Tips & Visualization Selection:

- **If target is categorical (classification):**
  - Count plot for class distribution
  - Check if balanced or imbalanced
- **If target is numeric (regression):**
  - Histogram for distribution
  - Check for skewness
- **If feature is numeric:**
  - Histogram for distribution
  - Box plot for outliers
  - Scatter plot vs target
- **If feature is categorical:**
  - Count plot for frequencies
  - Box plot (categorical vs numeric target)
  - Grouped bar plot
- **For relationships:**
  - Correlation heatmap (numeric-numeric)
  - Scatter plot (numeric-numeric)
  - Box plot (categorical-numeric)
  - Grouped bar (categorical-categorical)

### INSIGHTS TO WRITE:

- "The target variable is imbalanced with 70% class A and 30% class B, which may affect model performance."
- "Feature X shows a strong positive correlation (0.85) with the target, suggesting it's an important predictor."
- "There are significant outliers in the price column, with some values 10x higher than the median."
- "Age and income are highly correlated (0.92), we may consider dropping one to avoid multicollinearity."
- "Sales are significantly higher on weekends compared to weekdays."

---

# PHASE 2: DATA PREPROCESSING

## 🔢 **2.1 ENCODING CATEGORICAL VARIABLES**

### Sub-steps:

1. **Identify categorical columns**
2. **Choose encoding method**
3. **Apply encoding**
4. **Verify encoded data**

### What to Look For:

- Which columns are categorical?
- How many unique values? (cardinality)
- Is there an order/ranking? (ordinal vs nominal)
- Is this for the target or features?

### Encoding Methods & When to Use:

**LABEL ENCODING** (numbers: 0, 1, 2, 3...)

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])
# Example: ['red', 'blue', 'green'] → [0, 1, 2]
```

**WHEN TO USE:**

- ✅ For **target variable** in classification
- ✅ For **ordinal features** (small, medium, large)
- ❌ NOT for nominal features with ML (creates false order)

**ONE-HOT ENCODING** (binary columns: 0 or 1)

```python
# Pandas method
df = pd.get_dummies(df, columns=['category'], drop_first=True)

# Sklearn method
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(drop='first', sparse=False)
encoded = ohe.fit_transform(df[['category']])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out())
```

**WHEN TO USE:**

- ✅ For **nominal categorical features** (color, city, type)
- ✅ When feature has **< 10-15 unique values**
- ❌ NOT for high cardinality (creates too many columns)
- **drop_first=True** → Avoids multicollinearity

**ORDINAL ENCODING** (custom order)

```python
from sklearn.preprocessing import OrdinalEncoder

# Define order
size_order = [['small', 'medium', 'large', 'xl']]
ordinal_enc = OrdinalEncoder(categories=size_order)
df['size_encoded'] = ordinal_enc.fit_transform(df[['size']])
```

**WHEN TO USE:**

- ✅ For **ordinal features** with specific order
- ✅ Education level: High School < Bachelor < Master < PhD
- ✅ Ratings: Bad < OK < Good < Excellent

### WHY You Do It:

- **ML algorithms only understand numbers**, not text
- Different encoding methods preserve different information
- Wrong encoding can hurt model performance
- One-hot encoding prevents false ordinal relationships

### Tips:

- **If you see:** 2-10 categories → One-hot encoding
- **If you see:** > 15 categories → Consider target encoding or drop
- **If you see:** Target variable (for classification) → Label encoding
- **If you see:** Ordered categories → Ordinal encoding
- **If you see:** Binary (Yes/No) → Map to 0/1 directly
- Always encode AFTER train-test split or use fit on train, transform on test

---

## ⚖️ **2.2 FEATURE SCALING**

### Sub-steps:

1. **Identify numeric features**
2. **Choose scaling method**
3. **Apply scaling**
4. **Verify scaled data**

### What to Look For:

- Do numeric features have different ranges?
- Are there outliers?
- Which ML algorithm are you using?
- Is the distribution normal or skewed?

### Scaling Methods & When to Use:

**STANDARDIZATION (Standard Scaler)**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
# Result: Mean = 0, Std = 1
# Formula: (x - mean) / std
```

**WHEN TO USE:**

- ✅ Data has **outliers** (more robust than MinMax)
- ✅ Features have **different units** (age vs income)
- ✅ For: **Logistic Regression, SVM, KNN, Neural Networks**
- ✅ When data is **normally distributed**
- Default choice for most cases

**NORMALIZATION (MinMax Scaler)**

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
# Result: All values between 0 and 1
# Formula: (x - min) / (max - min)
```

**WHEN TO USE:**

- ✅ Need values in **specific range** [0, 1]
- ✅ Data has **no significant outliers**
- ✅ For: **Neural Networks, Image Processing**
- ❌ NOT good if you have outliers (they skew the range)

**ROBUST SCALER**

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
# Uses median and IQR instead of mean and std
```

**WHEN TO USE:**

- ✅ Data has **many outliers**
- ✅ More robust than StandardScaler
- ✅ When you want to **keep outliers** but reduce their impact

**NO SCALING NEEDED:**

- ❌ **Tree-based models** (Decision Tree, Random Forest, XGBoost)
- These models are scale-invariant

### WHY You Do It:

- **Distance-based algorithms** (KNN, SVM) need similar scales
- **Gradient-based algorithms** (Linear, Logistic, Neural Nets) converge faster
- Prevents features with large ranges from dominating
- Improves model performance and training speed

### Tips:

- **If you see:** KNN, SVM, Logistic Regression → MUST scale
- **If you see:** Random Forest, Decision Trees → DON'T scale
- **If you see:** Neural Networks → Scale (usually MinMax or Standard)
- **If you see:** Outliers → Use StandardScaler or RobustScaler
- **If you see:** No outliers → MinMaxScaler is fine
- Always fit scaler on training data only, then transform both train and test

---

## ⚖️ **2.3 HANDLING IMBALANCED DATA**

### Sub-steps:

1. **Check class distribution**
2. **Determine if it's imbalanced**
3. **Choose balancing method**
4. **Apply and verify**

### What to Look For:

- Is this a classification problem?
- What's the ratio between classes?
- How severe is the imbalance?

### Check Imbalance:

```python
# Check target distribution
df['target'].value_counts()
df['target'].value_counts(normalize=True)  # Percentage

# Visualize
df['target'].value_counts().plot(kind='bar')
```

**IMBALANCE SEVERITY:**

- Balanced: 40-60% split
- Slightly imbalanced: 20-40% minority class
- Moderately imbalanced: 10-20% minority class
- Severely imbalanced: < 10% minority class

### Balancing Methods:

**1. DO NOTHING** (sometimes best!)

```python
# Just make sure to use appropriate metrics
# Don't use accuracy, use: F1-score, Precision, Recall, AUC-ROC
```

**WHEN TO USE:**

- Slight imbalance (30-40% minority)
- Large dataset
- Tree-based models (they handle imbalance better)

**2. CLASS WEIGHTS** (easiest!)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Most sklearn models support this
model = LogisticRegression(class_weight='balanced')
model = RandomForestClassifier(class_weight='balanced')
```

**WHEN TO USE:**

- ✅ **FIRST CHOICE** for imbalanced data
- Works with most sklearn models
- No data manipulation needed
- Penalizes misclassifying minority class more

**3. OVERSAMPLING** (increase minority class)

```python
from imblearn.over_sampling import RandomOverSampler, SMOTE

# Random Oversampling (duplicate minority samples)
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# SMOTE (create synthetic samples)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

**WHEN TO USE:**

- Small dataset
- Severe imbalance
- **SMOTE is better** than random (creates new samples, not duplicates)
- ⚠️ Can lead to overfitting if overused

**4. UNDERSAMPLING** (reduce majority class)

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
```

**WHEN TO USE:**

- Large dataset (can afford to lose data)
- ⚠️ Loses information from majority class
- Usually **not recommended** unless dataset is huge

**5. COMBINED** (over + under)

```python
from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)
```

### WHY You Do It:

- **Imbalanced data** makes model biased toward majority class
- Model learns to predict majority class all the time
- Accuracy becomes misleading (99% accuracy but misses all minority!)
- Balancing helps model learn both classes properly

### Tips:

- **If you see:** 80-20 split → Try class_weight='balanced' first
- **If you see:** 95-5 split → Use SMOTE or class weights
- **If you see:** Very small dataset → Don't undersample!
- **Always balance AFTER train-test split**, only on training data
- **Don't balance test data** → Test should reflect real distribution

---

## 📊 **2.4 TRAIN-TEST SPLIT**

### Sub-steps:

1. **Separate features (X) and target (y)**
2. **Split into train and test sets**
3. **Verify split sizes**

### Common Code:

```python
from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop('target', axis=1)  # All columns except target
y = df['target']               # Only target column

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% for testing, 80% for training
    random_state=42,      # For reproducibility
    stratify=y            # Keeps class distribution in both sets
)

# Verify
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Train target distribution:\n{y_train.value_counts()}")
print(f"Test target distribution:\n{y_test.value_counts()}")
```

### Parameters:

- **test_size**: 0.2 (20%) or 0.3 (30%) are common
- **random_state**: Any number (42 is conventional) for reproducibility
- **stratify=y**: For classification, keeps same class ratio in train and test

### WHY You Do It:

- **Evaluate model on unseen data** → Measure true performance
- Prevents overfitting detection
- **Stratify** ensures both sets represent the data equally
- Industry standard practice

### Tips:

- **Always split BEFORE preprocessing** (except cleaning)
- **Use stratify=y** for classification problems
- **random_state** → Same results every time you run
- **If you see:** Small dataset (< 1000 rows) → Use 70-30 or cross-validation
- **If you see:** Large dataset → 80-20 split is fine

---

# PHASE 3: MODEL TRAINING

## 🤖 **3.1 CHOOSING ML ALGORITHMS**

### For CLASSIFICATION (Predict categories):

**LOGISTIC REGRESSION**

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
```

**WHEN TO USE:**

- Binary or multi-class classification
- ✅ Fast, simple, interpretable
- ✅ Good baseline model
- Works well with scaled data
- Need linear decision boundary

**DECISION TREE**

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42, max_depth=10)
model.fit(X_train, y_train)
```

**WHEN TO USE:**

- ✅ No scaling needed
- ✅ Handles non-linear relationships
- ✅ Easy to visualize and interpret
- ❌ Can overfit easily (control with max_depth)

**RANDOM FOREST**

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

**WHEN TO USE:**

- ✅ **One of the best all-around models**
- ✅ No scaling needed
- ✅ Handles non-linear relationships
- ✅ Reduces overfitting compared to Decision Tree
- ✅ Works well with imbalanced data (use class_weight='balanced')

**K-NEAREST NEIGHBORS (KNN)**

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
```

**WHEN TO USE:**

- ⚠️ **MUST scale data first**
- ✅ Simple, no training needed
- ✅ Works well with small datasets
- ❌ Slow on large datasets

**SUPPORT VECTOR MACHINE (SVM)**

```python
from sklearn.svm import SVC

model = SVC(kernel='rbf', random_state=42)
model.fit(X_train, y_train)
```

**WHEN TO USE:**

- ⚠️ **MUST scale data first**
- ✅ Good for high-dimensional data
- ✅ Effective with clear margin of separation
- ❌ Slow on large datasets

**NAIVE BAYES**

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
```

**WHEN TO USE:**

- ✅ Very fast
- ✅ Works well with text classification
- ✅ Good for small datasets

### For REGRESSION (Predict numbers):

**LINEAR REGRESSION**

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

**DECISION TREE REGRESSOR**

```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)
```

**RANDOM FOREST REGRESSOR**

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### SAFE 3-ALGORITHM COMBO FOR EXAM:

**For Classification:**

1. Logistic Regression (simple baseline)
2. Random Forest (robust, no scaling)
3. KNN or SVM (different approach, remember to scale!)

**For Regression:**

1. Linear Regression (simple baseline)
2. Random Forest Regressor (robust)
3. Decision Tree Regressor (interpretable)

---

## 🎯 **3.2 TRAINING MODELS**

### Standard Training Pattern:

```python
# 1. Import
from sklearn.linear_model import LogisticRegression

# 2. Instantiate
model = LogisticRegression(random_state=42)

# 3. Train
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)

# 5. Evaluate (see next section)
```

### Important Notes:

- **fit()** → Trains the model on training data
- **predict()** → Makes predictions on new data
- **random_state** → For reproducibility
- **Always train on X_train, y_train**
- **Always test on X_test** (NEVER on training data)

### WHY You Do It:

- Training teaches the model patterns in data
- Testing on separate data measures true performance
- Multiple algorithms give different perspectives
- Comparison helps choose best model

---

# PHASE 4: MODEL EVALUATION & COMPARISON

## 📊 **4.1 EVALUATION METRICS**

### For CLASSIFICATION:

**ACCURACY** (most common)

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Or directly from model
accuracy = model.score(X_test, y_test)
```

**WHEN TO USE:**

- ✅ Balanced datasets
- ❌ NOT for imbalanced data (misleading!)

**CONFUSION MATRIX**

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize
ConfusionMatrixDisplay(cm).plot()
plt.show()
```

**Shows:**

- True Positives, True Negatives
- False Positives, False Negatives

**PRECISION, RECALL, F1-SCORE**

```python
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

print(classification_report(y_test, y_pred))

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
```

**WHEN TO USE:**

- ✅ **Imbalanced data** (better than accuracy)
- **Precision**: Of predicted positives, how many are correct?
- **Recall**: Of actual positives, how many did we catch?
- **F1-Score**: Balance between Precision and Recall

### For REGRESSION:

**MEAN ABSOLUTE ERROR (MAE)**

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.4f}")
```

- Average absolute difference
- Easy to interpret

**MEAN SQUARED ERROR (MSE)**

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.4f}")
```

- Penalizes large errors more

**ROOT MEAN SQUARED ERROR (RMSE)**

```python
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")
```

- Same units as target variable
- Most commonly used

**R² SCORE (R-squared)**

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R²: {r2:.4f}")
```

- How much variance is explained
- Closer to 1.0 is better
- Can be negative if model is terrible

---

## 📊 **4.2 MODEL COMPARISON**

### Create Comparison Table:

```python
import pandas as pd

# Store results
results = {
    'Model': ['Logistic Regression', 'Random Forest', 'KNN'],
    'Accuracy': [0.85, 0.92, 0.88],
    'Precision': [0.84, 0.91, 0.87],
    'Recall': [0.86, 0.93, 0.89],
    'F1-Score': [0.85, 0.92, 0.88]
}

results_df = pd.DataFrame(results)
print(results_df)

# Visualize
results_df.set_index('Model').plot(kind='bar')
plt.title('Model Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
```

### Complete Workflow Example:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Dictionary to store models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}

# Train and evaluate each model
results = []

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)

    # Store
    results.append({
        'Model': name,
        'Accuracy': accuracy
    })

    # Print detailed report
    print(f"\n{name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

# Create comparison dataframe
comparison_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(comparison_df.sort_values('Accuracy', ascending=False))
```

### WHY You Do It:

- **Different models have different strengths**
- Some models work better on certain data types
- Comparison shows which performs best
- **Professors expect analysis**, not just numbers!

### INSIGHTS TO WRITE:

- "Random Forest achieved the highest accuracy (92%), outperforming Logistic Regression (85%) and KNN (88%)."
- "Despite lower accuracy, Logistic Regression showed better recall (0.86) for the minority class."
- "KNN required feature scaling while Random Forest did not, making RF more practical for this dataset."
- "All models showed similar F1-scores, suggesting consistent performance across classes."

---

# 🎯 EXAM DAY STRATEGY

## **WORKFLOW CHECKLIST:**

### STEP 1: Load & Inspect (5-10 min)

- [ ] Load dataset
- [ ] Check shape, columns, dtypes
- [ ] Identify target variable
- [ ] Identify feature types (numeric/categorical)
- [ ] Write insight: "Dataset overview"

### STEP 2: Data Cleaning (10-15 min)

- [ ] Check missing values
- [ ] Decide: drop or fill?
- [ ] Check duplicates
- [ ] Check for outliers
- [ ] Write insight: "What I cleaned and why"

### STEP 3: EDA & Visualization (15-20 min)

- [ ] Plot target distribution (check imbalance)
- [ ] Plot 2-3 important features
- [ ] Correlation heatmap
- [ ] 1-2 GroupBy aggregations
- [ ] Write insights for EACH visualization

### STEP 4: Feature Engineering (5-10 min)

- [ ] Create 1-2 new features (if applicable)
- [ ] Drop ID columns
- [ ] Write insight: "Why these features help"

### STEP 5: Preprocessing (10-15 min)

- [ ] Encode categorical variables
- [ ] Train-test split (80-20, stratify)
- [ ] Scale features (if needed for your models)
- [ ] Handle imbalance (if needed)
- [ ] Write insight: "Preprocessing choices"

### STEP 6: Model Training (15-20 min)

- [ ] Train Model 1 (e.g., Logistic Regression)
- [ ] Train Model 2 (e.g., Random Forest)
- [ ] Train Model 3 (e.g., KNN or SVM)
- [ ] Get predictions for each

### STEP 7: Evaluation (10-15 min)

- [ ] Calculate accuracy for each
- [ ] Print classification reports
- [ ] Create comparison table/plot
- [ ] Write insight: "Which model performed best and why"

### STEP 8: Final Review (5 min)

- [ ] Check all code cells run
- [ ] Check all insights are present
- [ ] Check visualizations have titles
- [ ] Verify final comparison

---

## ⚡ QUICK REFERENCE CHEAT SHEET

### IMPORTS (Copy this at the start):

```python
# Data handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Models - Classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Models - Regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Metrics
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Settings
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
```

### DECISION FLOWCHART:

**Missing Values?**

- < 5% → Drop rows
- 5-30% → Fill (mean/median/mode)
- > 30% → Drop column

**Categorical Feature?**

- Target variable → Label Encoding
- 2-10 categories → One-Hot Encoding
- Ordered categories → Ordinal Encoding

**Need Scaling?**

- Logistic, SVM, KNN → YES (StandardScaler)
- Random Forest, Decision Tree → NO

**Imbalanced Data?**

- Try class_weight='balanced' first
- If severe → SMOTE

**Which Metric?**

- Balanced classification → Accuracy
- Imbalanced classification → F1-Score
- Regression → RMSE or R²

---

## 🔥 COMMON MISTAKES TO AVOID

1. **Scaling before train-test split** → Data leakage!
2. **Not using random_state** → Results not reproducible
3. **Forgetting to encode categorical** → Error!
4. **Using accuracy on imbalanced data** → Misleading!
5. **Not writing insights** → Lost easy points!
6. **Scaling data for Random Forest** → Unnecessary
7. **Training on test data** → Cheating = Overfitting
8. **Balancing test data** → Wrong! Only balance training
9. **Not checking data shape after preprocessing** → Lose track
10. **Copying code without understanding** → Can't adapt in exam!

---

## 💡 POINTS-GRABBING TIPS

1. **Write insights at EVERY step** (easy points!)
2. **Label all plots** with titles, axis labels
3. **Use markdown cells** to organize sections
4. **Show comparison table** for models
5. **Mention why you chose each preprocessing step**
6. **Point out data characteristics** (imbalanced, skewed, outliers)
7. **Explain which model performed best and WHY**
8. **Use proper variable names** (not x, y, z everywhere)
9. **Format output nicely** (use print statements)
10. **Test your code runs from top to bottom** without errors

---

## END OF REVIEWER

**Good luck on your exam! 🚀**

Remember: Understanding > Memorization. Know the WHY, not just the HOW!

```

```
