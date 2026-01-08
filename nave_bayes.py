########## Nave Bayes ##########
########################################

# Importing the required libraries
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score

def execute(X_train, y_train, X_test, y_test):
  # Fitting the model to the training data
  model = GaussianNB()
  model = model.fit(X_train, y_train)

  # Evaluating the model performance using the test data
  y_pred = model.predict(X_test)

  result = metrics.confusion_matrix(y_test, y_pred)
  print("Confusion matrix:\n", result)

  target_names = ["without diabetes (0)", "with diabetes (1)"]
  print("\nClassification report:\n", classification_report(y_test, y_pred, target_names = target_names))

  r2 = r2_score(y_test, y_pred)
  print(f"R-squared: {r2:.2f}")

  mse = mean_squared_error(y_test, y_pred)
  print(f"Mean squared error: {mse:.2f}")

  rmse = mse ** 0.5
  print(f"Root mean squared error: {rmse:.2f}")
