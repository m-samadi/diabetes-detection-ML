########## K-Nearest Neighbors (KNN) ##########
########################################

# Importing the required libraries
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import RadiusNeighborsRegressor

def execute(dataset, X_train, y_train, X_test, y_test):
  ########## Unsupervised KNN Learning ##########
  print("\nUnsupervised KNN Learning")
  print("-------------------------")

  # Fitting the model to the training data
  model = NearestNeighbors(n_neighbors = 5, algorithm = 'ball_tree')
  model.fit(dataset)

  # Evaluating the model performance
  distances, indices = model.kneighbors(dataset)
  print("Indices:\n", indices)
  print("\nDistances:\n", distances)
  print("\nKneighbors graph:\n", model.kneighbors_graph(dataset).toarray())

  ########## Supervised KNN Learning (KNeighborsRegressor) ##########
  print("\nSupervised KNN Learning (KNeighborsRegressor)")
  print("---------------------------------------------")

  # Fitting the model to the training data
  model = KNeighborsRegressor(n_neighbors = 5)
  model.fit(X_train, y_train)

  # Evaluating the model performance using the test data
  y_pred = model.predict(X_test)

  r2 = r2_score(y_test, y_pred)
  print(f"R-squared: {r2:.2f}")

  mse = mean_squared_error(y_test, y_pred)
  print(f"Mean squared error: {mse:.2f}")

  rmse = mse ** 0.5
  print(f"Root mean squared error: {rmse:.2f}")

  ########## Supervised KNN Learning (RadiusNeighborsRegressor) ##########
  print("\nSupervised KNN Learning (RadiusNeighborsRegressor)")
  print("--------------------------------------------------")

  # Fitting the model to the training data
  model = RadiusNeighborsRegressor(radius = 1)
  model.fit(X_train, y_train)

  # Evaluating the model performance using the test data
  y_pred = model.predict(X_test)

  r2 = r2_score(y_test, y_pred)
  print(f"R-squared: {r2:.2f}")

  mse = mean_squared_error(y_test, y_pred)
  print(f"Mean squared error: {mse:.2f}")

  rmse = mse ** 0.5
  print(f"Root mean squared error: {rmse:.2f}")
