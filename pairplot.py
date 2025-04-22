import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read data from CSV file
df = pd.read_csv('soil_fertility_output.csv')  # Replace 'soil_data.csv' with your actual file name

# Select only the required nutrient columns
nutrients = ['N', 'P', 'K', 'S', 'Zn', 'Cu', 'Fe', 'Mn', 'B']
df = df[nutrients]

# User selects which nutrient to visualize
print("Available nutrients:", nutrients)
selected_nutrient = input("Enter the nutrient to visualize pairwise relationships: ")

if selected_nutrient in nutrients:
    g = sns.pairplot(df, hue=selected_nutrient, diag_kind='kde', corner=True)
    g.fig.suptitle(f"Pairwise Scatter Plots for {selected_nutrient}", y=1.02)
    g.map_lower(sns.scatterplot)
    g.map_diag(sns.kdeplot)
    plt.show()
else:
    print("Invalid nutrient selection. Please choose from the list above.")
