import numpy as np
import pandas as pd
import joblib
from scipy.optimize import linprog

# Function to make predictions using the trained model
def predict_fertility(input_data):
    """
    Predict fertility value based on input data.
    :param input_data: DataFrame containing new data with columns ['N', 'P', 'K', 'S', 'Zn', 'Cu', 'Fe', 'Mn', 'B']
    :return: Predicted fertility values
    """
    # Load the trained model and scaler
    loaded_model = joblib.load("fertility_model.pkl")
    loaded_scaler = joblib.load("scaler.pkl")
    
    # Transform input data
    input_scaled = loaded_scaler.transform(input_data)
    
    # Make prediction
    predictions = loaded_model.predict(input_scaled)
    return predictions

new_data = pd.DataFrame({       #change values as per target land features
    'N': [55.16],
    'P': [28.0],
    'K': [100],
    'S': [5.78],
    'Zn': [0.73],
    'Cu': [0.7],
    'Fe': [2.64],
    'Mn': [2.36],
    'B': [0.3]
})

# Predict fertility values
predictions = predict_fertility(new_data)
print(predictions)

# Function to get the maximum profit for each combination of crops
def max_profit(total_area, crop_data):
    num_crops = len(crop_data)
    # Extract crop market prices and yields from the data
    prices = np.array([crop['price'] for crop in crop_data])
    yields = np.array([crop['yield'] for crop in crop_data])
    
    # Define the objective coefficients (profit per hectare for each crop)
    c = -prices * yields  # We negate because linprog minimizes
    
    # Bounds for each crop (total area per crop must be between 0 and total_area)
    bounds = [(0, total_area) for _ in range(num_crops)]
    
    # Constraints: the sum of areas allocated to the crops must equal the total area
    A_eq = np.ones((1, num_crops))
    b_eq = np.array([total_area])
    
    # Run linear programming to maximize profit
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if result.success:
        area_allocation = result.x
        total_profit = -result.fun  # Negate the profit back
        return area_allocation, total_profit
    else:
        print("Optimization failed.")
        return None, 0

# Main function to input data and compute results
def main():
    # Input data for the crops (price per kg and yield per hectare)
    crop_data = []
    crop_names = ["Bajra", "Wheat", "Mustard", "Barley"]
    MSP = [26.25, 24.25, 59.50, 19.80]
    COP = [14.85, 11.82, 30.11, 12.39]
    market_price = [a - b for a, b in zip(MSP, COP)]     #change data as per market values
    for i in range(4):
        crop_data.append({'name': crop_names[i], 'price': market_price[i], 'yield': predictions[0, i]})
    
    total_area = float(input("Enter the total available area (hectares): "))
    
    # Calculate the optimal combinations for 2, 3, and 4 crops
    print("\nCalculating optimal area allocations for combinations of crops...\n")
    
    for num_crops in [2, 3, 4]:
        print(f"\nFor {num_crops} crops:")
        combinations = []
        max_profit_combination = None
        max_profit_value = 0
        
        # Generate all combinations of crops for the selected number
        for combination in itertools.combinations(crop_data, num_crops):
            area_allocation, total_profit = max_profit(total_area, combination)
            if total_profit > max_profit_value:
                max_profit_value = total_profit
                max_profit_combination = area_allocation
                combinations = combination
        
        print(f"Best combination: {', '.join([crop['name'] for crop in combinations])}")
        print(f"Optimal area allocation: {max_profit_combination}")
        print(f"Total profit: {max_profit_value}")

# Run the program
if __name__ == "__main__":
    import itertools
    main()