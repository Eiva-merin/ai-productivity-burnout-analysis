# ai-productivity-burnout-analysis

# AI Productivity & Employee Burnout Risk Analysis

A comprehensive ML portfolio project analysing how AI tool usage 
affects employee productivity and predicting burnout risk using 
classical and deep learning models on 4,500 employee records.

## Dataset
4,500 employee records with 17 features including AI tool usage hours,
tasks automated, manual work hours, deadline pressure, focus hours,
work-life balance score and burnout risk score.

## Project Structure — 8 Parts

| Part | Topic |
|------|-------|
| 1 | Data loading, EDA and visualisation |
| 2 | Regression — predicting productivity score |
| 3 | Classification — predicting burnout risk level |
| 4 | Advanced models — Random Forest, SVM |
| 5 | Neural Network (Keras) for classification |
| 6 | Unsupervised learning — K-Means clustering |
| 7 | Model comparison and evaluation |
| 8 | Ethics, fairness and explainability |

## Models Used
- Linear Regression, Decision Tree, k-NN
- Logistic Regression, Random Forest, SVM
- Neural Network (Keras MLP)
- K-Means Clustering
- SMOTE for class imbalance handling

## Key Results
- Best classification accuracy: ~99% (Random Forest + Neural Network)
- Logistic Regression: best balance of accuracy and interpretability
- K-Means revealed distinct employee productivity clusters

## Tech Stack
Python · pandas · NumPy · scikit-learn · TensorFlow/Keras · 
Matplotlib · Seaborn · imbalanced-learn

## How to Run
pip install -r requirements.txt
jupyter notebook ai_productivity_burnout_analysis.ipynb

## Self-Learning Project
Built independently to demonstrate end-to-end ML skills across
regression, classification, clustering, deep learning and 
responsible AI practices.