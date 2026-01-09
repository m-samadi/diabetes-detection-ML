# Diabetes detection using machine learning (ML) algorithms
## Introduction
This repository includes an evaluation of different ML algorithms for diabetes detection in Scikit-Learn. The analysis was performed using a dataset originally provided by the National Institute of Diabetes and Digestive and Kidney Diseases, available at the following link:
```
https://www.kaggle.com/datasets/mathchi/diabetes-data-set/
```
## Evaluation
The evaluations based on the current configurations performed on the algorithms are summarized in the table below, indicating that the three algorithms, including Regression With Random Forest, Regression With Extra-Tree, and LinearSVR, obtained higher accuracy (0.83) compared to the others.
<br/><br/>

| Algorithm | Accuracy |
| :---: | :---: |
| K-Means Clustering | 0.68 |
| Linear Regression | 0.70 |
| Decision Tree - Classification With Decision Tree | 0.66 |
| Decision Tree - Regression With Decision Tree | 0.69 |
| Randomized Decision Tree - Classification With Random Forest | 0.73 |
| Randomized Decision Tree - Regression With Random Forest | **0.83** |
| Randomized Decision Tree - Classification With Extra-Tree | 0.75 |
| Randomized Decision Tree - Regression With Extra-Tree | **0.83** |
| Logistic Regression | 0.70 |
| Support Vector Machine (SVM) - Support Vector Classifier (SVC) | 0.70 |
| Support Vector Machine (SVM) - Nu Support Vector Classification (NuSVC) | 0.70 |
| Support Vector Machine (SVM) - Linear Support Vector Classification (LinearSVC) | 0.69 |
| Support Vector Machine (SVM) - Support Vector Regression (SVR) | 0.82 |
| Support Vector Machine (SVM) - Nu Support Vector Regression (NuSVR) | 0.75 |
| Support Vector Machine (SVM) - Linear Support Vector Regression (LinearSVR) | **0.83** |
| Nave Bayes | 0.69 |
| K-Nearest Neighbors (KNN) - Unsupervised KNN Learning | - |
| K-Nearest Neighbors (KNN) - Supervised KNN Learning (KNeighborsRegressor) | 0.80 |
| K-Nearest Neighbors (KNN) - Supervised KNN Learning (RadiusNeighborsRegressor) | 0.78 |

<br/><br/>
It is worth noting that the assessments can also be performed using other available datasets, applying different features and configurations for diabetes detection.
## Program execution
First, the main program can be executed using the code below:
```
python main.py
```
Next, one of the ML algorithms should be selected in the terminal by entering a number between 1 and 7.
