import pandas as pd
from scipy.sparse import hstack
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import manual_bow

df = pd.read_csv("data/manual-incomplete.csv")

# Define function to load embeddings
def id2filename(i):
    return f"data/man-embeddings/{i}.pt"

# Initialize lists to store embeddings and labels
embeddings = []
labels = []

# Iterate over each row in the DataFrame
for i, row in df.iterrows():
    id = row["id"]
    mean_thankfulness = row["mean thankfulness"]
    mean_animosity = row["mean animosity"]
    
    # Load embedding
    embedding = torch.load(id2filename(id))
    embeddings.append(embedding.numpy())
    
    # Append labels
    labels.append([mean_thankfulness, mean_animosity])

# Convert lists to numpy arrays
X_bert = np.array(embeddings)
y = np.array(labels)
X_tf = manual_bow.X_train_tf
X = hstack([X_bert, X_tf, manual_bow.X_train_bow])

# Now you have X and y, you can proceed with training your model
# For example, splitting data into train and test sets and training a linear model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

from sklearn.dummy import DummyRegressor

# Create a dummy regressor that predicts the mean of the target variable
dummy_regressor = DummyRegressor(strategy='mean')

# Fit the dummy regressor on the training data (not necessary for mean strategy)
dummy_regressor.fit(X_train, y_train)

# Make predictions using the dummy regressor
dummy_predictions = dummy_regressor.predict(X_test)

# Compute metrics for the dummy regressor
dummy_mae = mean_absolute_error(y_test, dummy_predictions)
dummy_mse = mean_squared_error(y_test, dummy_predictions)
dummy_r_squared = r2_score(y_test, dummy_predictions)

# Print the scores of the dummy regressor
print("Dummy Regressor Scores:")
print("Mean Absolute Error (MAE):", dummy_mae)
print("Mean Squared Error (MSE):", dummy_mse)
print("R-squared (R2):", dummy_r_squared)


for alpha in [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.1, 1, 10, 100, 1000]:
    # Initialize linear regression model
    model = Lasso(alpha=alpha)
    
    # Train the model
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    # Compute metrics
    for var in 0, 1:
        mask = slice(None), var
        mae = mean_absolute_error(y_test[mask], y_pred[mask])
        mse = mean_squared_error(y_test[mask], y_pred[mask])
        r_squared = r2_score(y_test[mask], y_pred[mask])
        
        # Print the metrics
        print(alpha, "Mean Absolute Error (MAE):", mae)
        print(alpha, "Mean Squared Error (MSE):", mse)
        print(alpha, "R-squared (R2):", r_squared)

    y_pred = model.predict(X)
    df[f"y-pred-thankfulness-{alpha}"] = y_pred[:,0]
    df[f"y-pred-animosity-{alpha}"] = y_pred[:,1]

    print(model.coef_ != 0)

df.to_csv("tune_alpha.csv")
