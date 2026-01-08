########## Support Vector Machine (SVM) ##########
########################################

# Importing the required libraries
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVR
from sklearn.svm import LinearSVR

def execute(X_train, y_train, X_test, y_test):
  ########## Support Vector Classifier (SVC) ##########
  print("\nSupport Vector Classifier (SVC)")
  print("-------------------------------")

  # Fitting the model to the training data
  model = svm.SVC(kernel = 'linear')
  model.fit(X_train, y_train)

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

  print("Intercept:", model.intercept_)

  print("Fit status:", model.fit_status_)

  print("\nCoefficients:\n", model.coef_)

  print("\nNumber of support vectors for each class:\n", model.n_support_)

  print("\nSupport vectors:\n", model.support_vectors_)

  print("\nIndices of support vectors:\n", model.support_)

  ########## Nu Support Vector Classification (NuSVC) ##########
  print("\nNu Support Vector Classification (NuSVC)")
  print("----------------------------------------")

  # Fitting the model to the training data
  model = NuSVC(kernel = 'linear', gamma = 'scale', shrinking = False)
  model.fit(X_train, y_train)

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

  print("Intercept:", model.intercept_)

  print("Fit status:", model.fit_status_)

  print("\nCoefficients:\n", model.coef_)

  print("\nNumber of support vectors for each class:\n", model.n_support_)

  print("\nSupport vectors:\n", model.support_vectors_)

  print("\nIndices of support vectors:\n", model.support_)

  ########## Linear Support Vector Classification (LinearSVC) ##########
  print("\nLinear Support Vector Classification (LinearSVC)")
  print("------------------------------------------------")

  # Fitting the model to the training data
  model = LinearSVC(dual = False, random_state = 42, penalty = 'l1', tol = 1e-5)
  model.fit(X_train, y_train)

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

  print("Intercept:", model.intercept_)

  print("\nCoefficients:\n", model.coef_)

  ########## Support Vector Regression (SVR) ##########
  print("\nSupport Vector Regression (SVR)")
  print("-------------------------------")

  # Fitting the model to the training data
  model = svm.SVR(kernel = 'linear', gamma = 'auto')
  model = model.fit(X_train, y_train)

  # Evaluating the model performance using the test data
  y_pred = model.predict(X_test)

  r2 = r2_score(y_test, y_pred)
  print(f"R-squared: {r2:.2f}")

  mse = mean_squared_error(y_test, y_pred)
  print(f"Mean squared error: {mse:.2f}")

  rmse = mse ** 0.5
  print(f"Root mean squared error: {rmse:.2f}")

  print("Intercept:", model.intercept_)

  print("Fit status:", model.fit_status_)

  print("\nCoefficients:\n", model.coef_)

  print("\nNumber of support vectors for each class:\n", model.n_support_)

  print("\nSupport vectors:\n", model.support_vectors_)

  print("\nIndices of support vectors:\n", model.support_)

  ########## Nu Support Vector Regression (NuSVR) ##########
  print("\nNu Support Vector Regression (NuSVR)")
  print("------------------------------------")

  # Fitting the model to the training data
  model = NuSVR(kernel = 'linear', gamma = 'auto', C = 1.0, nu = 0.1)
  model.fit(X_train, y_train)

  # Evaluating the model performance using the test data
  y_pred = model.predict(X_test)

  r2 = r2_score(y_test, y_pred)
  print(f"R-squared: {r2:.2f}")

  mse = mean_squared_error(y_test, y_pred)
  print(f"Mean squared error: {mse:.2f}")

  rmse = mse ** 0.5
  print(f"Root mean squared error: {rmse:.2f}")

  print("Intercept:", model.intercept_)

  print("Fit status:", model.fit_status_)

  print("\nCoefficients:\n", model.coef_)

  print("\nNumber of support vectors for each class:\n", model.n_support_)

  print("\nSupport vectors:\n", model.support_vectors_)

  print("\nIndices of support vectors:\n", model.support_)

  ########## Linear Support Vector Regression (LinearSVR) ##########
  print("\nLinear Support Vector Regression (LinearSVR)")
  print("--------------------------------------------")

  # Fitting the model to the training data
  model = LinearSVR(dual = False, random_state = 42, loss = 'squared_epsilon_insensitive', tol = 1e-5)
  model.fit(X_train, y_train)

  # Evaluating the model performance using the test data
  y_pred = model.predict(X_test)

  r2 = r2_score(y_test, y_pred)
  print(f"R-squared: {r2:.2f}")

  mse = mean_squared_error(y_test, y_pred)
  print(f"Mean squared error: {mse:.2f}")

  rmse = mse ** 0.5
  print(f"Root mean squared error: {rmse:.2f}")

  print("Intercept:", model.intercept_)

  print("\nCoefficients:\n", model.coef_)
