import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("fertility_data.csv")

# Separate features and target variable
X = data[['N', 'P', 'K', 'S', 'Zn', 'Cu', 'Fe', 'Mn', 'B']]
y = data[['FV', 'BajraY', 'WheatY', 'MustardY', 'BarleyY']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Save the trained model and scaler
joblib.dump(model, "fertility_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully.")