# Importing the required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Importing the external source files
import kmeans
import linear_regression
import decision_tree
import logistic_regression
import svm
import nave_bayes
import knn

########## Importing the dataset ##########
dataset = pd.read_csv("dataset/diabetes.csv")

########## Performing data visualization ##########
print("\nDataset:\n", dataset.head(10))
print("\nDataset dimensions:", dataset.shape)
print("\nData categorization:\n", dataset.groupby("Outcome").size())

########## Performing data pre-processing ##########
manipulated_dataset = dataset
# Replacing zero values with NaN
manipulated_dataset[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = manipulated_dataset[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.nan)

# Replacing NaN values with mean values
manipulated_dataset.fillna({"Glucose": manipulated_dataset["Glucose"].mean()}, inplace = True)
manipulated_dataset.fillna({"BloodPressure": manipulated_dataset["BloodPressure"].mean()}, inplace = True)
manipulated_dataset.fillna({"SkinThickness": manipulated_dataset["SkinThickness"].mean()}, inplace = True)
manipulated_dataset.fillna({"Insulin": manipulated_dataset["Insulin"].mean()}, inplace = True)
manipulated_dataset.fillna({"BMI": manipulated_dataset["BMI"].mean()}, inplace = True)

# Performing the feature scaling process using MinMaxScaler in the range (0, 1)
scale = MinMaxScaler(feature_range = (0, 1))
scaled_data = scale.fit_transform(manipulated_dataset)
scaled_dataset = pd.DataFrame(scaled_data)
print("\nScaled data:\n", scaled_dataset)

########## Doing feature importance and visualizing the clusters ##########
# Displaying the heatmap
sns.heatmap(scaled_dataset.corr(), annot = True)
plt.savefig("output/heatmap.png")

# Displaying the detailed distribution of the features
sns.pairplot(data = dataset, hue = "Outcome")
plt.savefig("output/distribution.png")

########## Fitting and evaluating the model using the selected algorithm ##########
X = scaled_dataset.iloc[:, 0:8].values
y = scaled_dataset.iloc[:, 8].values

# Splitting X and y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42, stratify = manipulated_dataset["Outcome"])

# Displaying the dimensions of the selected data
print("\nX_train's dimensions:", X_train.shape)
print("X_test's dimensions:", X_test.shape)
print("y_train's dimensions:", y_train.shape)
print("y_test's dimensions:", y_test.shape)

# Selecting the algorithm
print("\n1: K-Means Clustering")
print("2: Linear Regression")
print("3: Decision Tree")
print("4: Logistic Regression")
print("5: Support Vector Machine (SVM)")
print("6: Nave Bayes")
print("7: K-Nearest Neighbors (KNN)")
id = input("Enter the ID of the selected algorithm: ")

match int(id):
  case 1:
    kmeans.execute(X_train, X_test, y_test)
  case 2:
    linear_regression.execute(dataset, X_train, y_train, X_test, y_test)
  case 3:
    decision_tree.execute(X, y, X_train, y_train, X_test, y_test)
  case 4:
    logistic_regression.execute(X_train, y_train, X_test, y_test)
  case 5:
    svm.execute(X_train, y_train, X_test, y_test)
  case 6:
    nave_bayes.execute(X_train, y_train, X_test, y_test)
  case 7:
    knn.execute(dataset, X_train, y_train, X_test, y_test)
  case _:
    print("The ID entered is incorrect.")
