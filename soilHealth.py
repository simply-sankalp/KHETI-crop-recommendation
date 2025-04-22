import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define nutrient sufficiency ranges (example values in mg/kg)
sufficiency_ranges = {
    "N": (40, 60), "P": (20, 40), "K": (150, 250), "S": (10, 30),
    "Zn": (1, 5), "Cu": (0.2, 1.5), "Fe": (2, 10), "Mn": (1, 6), "B": (0.5, 2.5)
}

# Define nutrient weights
weights = {
    "N": 0.2, "P": 0.15, "K": 0.15, "S": 0.1,
    "Zn": 0.1, "Cu": 0.07, "Fe": 0.08, "Mn": 0.08, "B": 0.07
}

# Function to normalize nutrient values to a 0-100 scale
def normalize_score(value, min_val, max_val):
    if value >= max_val:
        return 100
    elif value <= min_val:
        return 0
    else:
        return ((value - min_val) / (max_val - min_val)) * 100

# Function to compute Soil Fertility Index (SFI)
def compute_sfi(soil_sample):
    scores = {}
    for nutrient, value in soil_sample.items():
        min_val, max_val = sufficiency_ranges[nutrient]
        scores[nutrient] = normalize_score(value, min_val, max_val)
    
    sfi = sum(weights[n] * scores[n] for n in weights) / sum(weights.values())
    return sfi, scores

# Read soil data from CSV file and compute SFI
def process_csv(file_path):
    df = pd.read_csv(file_path)
    sfi_values = []
    
    for index, row in df.iterrows():
        soil_sample = {nutrient: row[nutrient] for nutrient in sufficiency_ranges.keys()}
        sfi_score, _ = compute_sfi(soil_sample)
        sfi_values.append(sfi_score)
    
    df["SFI"] = sfi_values
    df.to_csv("soil_fertility_output.csv", index=False)
    print("Processed data saved to soil_fertility_output.csv")
    
    # Plot distribution of SFI values
    plt.hist(sfi_values, bins=range(0, 110, 10), edgecolor='black', alpha=0.7)
    plt.xlabel("SFI Range")
    plt.ylabel("Number of Data Points")
    plt.title("Distribution of Soil Fertility Index (SFI)")
    plt.xticks(range(0, 110, 10))
    plt.show()
    
    return df

# Read yield data from CSV and plot against SFI
def plot_yield_vs_sfi(file_path):
    df = pd.read_csv(file_path)
    if "SFI" not in df.columns:
        print("SFI values not found in the dataset. Computing SFI...")
        df = process_csv(file_path)
    
    crop_columns = ["BajraY", "WheatY", "MustardY", "BarleyY"]
    
    # Subplots for individual yield vs. SFI plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    for i, crop in enumerate(crop_columns):
        axes[i].scatter(df["SFI"], df[crop], alpha=0.7, label=crop)
        axes[i].set_xlabel("Soil Fertility Index (SFI)")
        axes[i].set_ylabel(f"{crop} Yield")
        axes[i].set_title(f"{crop} Yield vs. SFI")
        axes[i].grid(True)
        axes[i].legend()
    plt.tight_layout()
    plt.show()
    
    # Superimposed plot with x-axis limited to 1-50
    plt.figure(figsize=(8, 6))
    for crop in crop_columns:
        plt.scatter(df["SFI"], df[crop], alpha=0.7, label=crop)
    plt.xlabel("Soil Fertility Index (SFI)")
    plt.ylabel("Yield")
    plt.title("Yield vs. Soil Fertility Index (Superimposed)")
    plt.xlim(1, 100)
    plt.grid(True)
    plt.legend()
    plt.show()

# # Example soil test data (concentration values in mg/kg)
# soil_sample = {
#     "N": 55, "P": 30, "K": 180, "S": 15,
#     "Zn": 3, "Cu": 0.8, "Fe": 5, "Mn": 2, "B": 1.5
# }

# # Compute SFI
# sfi_score, nutrient_scores = compute_sfi(soil_sample)

# # Display results
# print("Nutrient Scores:", nutrient_scores)
# print(f"Soil Fertility Index (SFI): {sfi_score:.2f}")

# # Visualization
# plt.bar(nutrient_scores.keys(), nutrient_scores.values(), color='skyblue')
# plt.axhline(y=sfi_score, color='r', linestyle='dashed', label=f'SFI: {sfi_score:.2f}')
# plt.xlabel("Nutrients")
# plt.ylabel("Score (0-100)")
# plt.title("Soil Fertility Index & Nutrient Scores")
# plt.legend()
# plt.show()

# Example usage for CSV file
# process_csv("soil_data.csv")
plot_yield_vs_sfi("soil_fertility_output.csv")
