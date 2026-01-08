########## Decision Tree ##########
########################################

# Importing the required libraries
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesRegressor

def execute(X, y, X_train, y_train, X_test, y_test):
  ########## Classification With Decision Tree ##########
  print("\nClassification With Decision Tree")
  print("---------------------------------")

  # Fitting the model to the training data
  model = tree.DecisionTreeClassifier()
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

  ########## Regression With Decision Tree ##########
  print("\nRegression With Decision Tree")
  print("-----------------------------")

  # Fitting the model to the training data
  model = tree.DecisionTreeRegressor()
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

  ########## Classification With Random Forest ##########
  print("\nClassification With Random Forest")
  print("---------------------------------")

  # Fitting the model to the training data
  model = RandomForestClassifier(n_estimators = 100)
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

  ########## Regression With Random Forest ##########
  print("\nRegression With Random Forest")
  print("-----------------------------")

  # Fitting the model to the training data
  model = RandomForestRegressor(max_depth = 20, random_state = 42, n_estimators = 100)
  model = model.fit(X_train, y_train)

  # Evaluating the model performance using the test data
  y_pred = model.predict(X_test)

  r2 = r2_score(y_test, y_pred)
  print(f"R-squared: {r2:.2f}")

  mse = mean_squared_error(y_test, y_pred)
  print(f"Mean squared error: {mse:.2f}")

  rmse = mse ** 0.5
  print(f"Root mean squared error: {rmse:.2f}")

  ########## Classification With Extra-Tree ##########
  print("\nClassification With Extra-Tree")
  print("------------------------------")

  # Fitting the model to the training data
  kfold = KFold(n_splits = 10)
  model = ExtraTreesClassifier(n_estimators = 100, max_features = 5)
  scores = cross_val_score(model, X, y, cv = kfold).mean()

  # Evaluating the model performance
  print(f"Scores: {scores:.2f}")

  ########## Regression With Extra-Tree ##########
  print("\nRegression With Extra-Tree")
  print("--------------------------")

  # Fitting the model to the training data
  model = ExtraTreesRegressor(max_depth = 20, random_state = 42, n_estimators = 100)
  model.fit(X_train, y_train)

  # Evaluating the model performance using the test data
  y_pred = model.predict(X_test)

  r2 = r2_score(y_test, y_pred)
  print(f"R-squared: {r2:.2f}")

  mse = mean_squared_error(y_test, y_pred)
  print(f"Mean squared error: {mse:.2f}")

  rmse = mse ** 0.5
  print(f"Root mean squared error: {rmse:.2f}")
