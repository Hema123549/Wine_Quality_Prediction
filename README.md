# Wine Quality Prediction 🍷

## Project Overview
This project predicts whether a wine is of good or bad quality using machine learning classification algorithms.

The project uses the Wine Quality dataset containing red and white wine samples. Two machine learning models, Logistic Regression and Random Forest, are trained and evaluated.

## Dataset
The dataset contains two files:
- winequality-red.csv
- winequality-white.csv

Dataset statistics:
- Red wine samples: 1,599
- White wine samples: 4,898
- Total samples: 6,497
- Features: 11 physicochemical properties

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- VS Code

## Project Structure
```text
Wine_Quality_Prediction/
│
├── dataset/
│   ├── winequality-red.csv
│   └── winequality-white.csv
│
├── src/
│   ├── data_analysis.py
│   ├── eda.py
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── visualize_results.py
│   └── predict_wine.py
│
├── outputs/
│   ├── combined_wine.csv
│   ├── prepared_wine.csv
│   ├── quality_distribution.png
│   ├── alcohol_vs_quality.png
│   ├── volatile_acidity_vs_quality.png
│   ├── wine_type_distribution.png
│   ├── wine_correlation_heatmap.png
│   ├── model_accuracy_comparison.png
│   ├── random_forest_confusion_matrix.png
│   ├── logistic_confusion_matrix.png
│   ├── logistic_model.pkl
│   └── random_forest_model.pkl
│
├── README.md
└── requirements.txt
```

## Project Workflow

### 1. Data Collection
Loaded red and white wine datasets and combined them into a single dataset.

### 2. Data Analysis
- Checked dataset dimensions.
- Checked missing values.
- Analyzed wine quality distribution.
- Compared red and white wine samples.

### 3. Exploratory Data Analysis
Created visualizations to understand:
- Wine quality distribution.
- Alcohol vs. wine quality.
- Volatile acidity vs. wine quality.
- Red and white wine distribution.
- Feature correlations.

### 4. Data Preparation
- Combined both datasets.
- Created a binary target variable called `quality_label`.
- Assigned 1 to wines with quality >= 7.
- Assigned 0 to wines with quality < 7.
- Applied one-hot encoding to the wine type column.

### 5. Model Training
Trained two classification algorithms:

- Logistic Regression
- Random Forest Classifier

Used an 80:20 train-test split with stratification.

### 6. Model Evaluation

Compared models using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

### Model Performance

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 71.23% | 88.54% |
| Precision | 38.25% | 80.92% |
| Recall | 75.00% | 54.69% |
| F1 Score | 50.66% | 65.27% |
| ROC-AUC | 80.19% | 91.47% |

### 7. Sample Prediction
Tested the trained Random Forest model using a wine sample from the dataset.

Example prediction:
- Predicted Quality: Good Quality
- Probability of Bad Quality: 34.67%
- Probability of Good Quality: 65.33%

## Conclusion
The Random Forest model achieved better overall performance than Logistic Regression, with an accuracy of 88.54% and ROC-AUC score of 91.47%.

Logistic Regression achieved higher recall for the good-quality class, while Random Forest provided better precision and F1 score.

Therefore, Random Forest was selected as the primary model for this project.

## Note
The target classification uses a chosen quality threshold of 7. Predictions are model estimates based on the dataset and should not be treated as professional wine-quality assessments.