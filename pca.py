import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def perform_pca_for_crop(data, crop_name, nutrients):
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[nutrients])
    
    # Perform PCA
    pca = PCA(n_components=2)  # Reduce to 2D for visualization
    principal_components = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    
    # PCA Loadings (to check redundancy)
    loadings = pd.DataFrame(pca.components_.T, index=nutrients, 
                            columns=['PC1', 'PC2'])
    
    return explained_variance, loadings, principal_components, pca.components_

def plot_pca_results(crops, explained_variances):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, crop in enumerate(crops):
        sns.barplot(x=np.arange(1, len(explained_variances[i]) + 1), 
                    y=explained_variances[i], palette='viridis', ax=axes[i])
        axes[i].set_xlabel('Principal Components')
        axes[i].set_ylabel('Explained Variance Ratio')
        axes[i].set_title(f'PCA Explained Variance for {crop}')
    
    plt.tight_layout()
    plt.show()

def plot_pca_clusters(crops, pca_results, df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, crop in enumerate(crops):
        pc_data = pca_results[i]
        scatter = axes[i].scatter(pc_data[:, 0], pc_data[:, 1], c=df[crop], cmap='viridis', edgecolors='k', alpha=0.7)
        axes[i].set_xlabel('PC1')
        axes[i].set_ylabel('PC2')
        axes[i].set_title(f'PCA Scatter Plot for {crop}')
        fig.colorbar(scatter, ax=axes[i], label='Yield')
    
    plt.tight_layout()
    plt.show()

def plot_pca_clusters_with_vectors(crops, pca_results, df, loadings, nutrients):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, crop in enumerate(crops):
        pc_data = pca_results[i]
        scatter = axes[i].scatter(pc_data[:, 0], pc_data[:, 1], c=df[crop], cmap='viridis', edgecolors='k', alpha=0.7)
        axes[i].set_xlabel('PC1')
        axes[i].set_ylabel('PC2')
        axes[i].set_title(f'PCA Scatter Plot for {crop}')
        fig.colorbar(scatter, ax=axes[i], label='Yield')
        
        # Plot PCA Loadings as vectors
        for j, nutrient in enumerate(nutrients):
            axes[i].arrow(0, 0, loadings[i][0, j] * 2, loadings[i][1, j] * 2, color='red', alpha=0.75, head_width=0.1)
            axes[i].text(loadings[i][0, j] * 2.2, loadings[i][1, j] * 2.2, nutrient, color='red', fontsize=10)
    
    plt.tight_layout()
    plt.show()

# Load dataset
df = pd.read_csv("soil_fertility_output.csv")  # Replace with actual file path

# Define nutrient columns
nutrients = ['N', 'P', 'K', 'S', 'Zn', 'Cu', 'Fe', 'Mn', 'B']

# Define crop names
crops = ['BajraY', 'WheatY', 'MustardY', 'BarleyY']

# Perform PCA and collect results
explained_variances = []
pca_results = []
redundant_features_list = []
loadings_list = []

for crop in crops:
    print(f"\nPerforming PCA for {crop} Yield")
    explained_variance, loadings, principal_components, components = perform_pca_for_crop(df, crop, nutrients)
    explained_variances.append(explained_variance)
    pca_results.append(principal_components)
    loadings_list.append(components)
    
    # Identify redundant parameters (features with high loadings in the same PC)
    threshold = 0.6  # Consider absolute values > 0.6 as significant loadings
    redundant_features = []
    for i in range(loadings.shape[1]):  # Iterate over available PCs (PC1, PC2)
        high_loadings = loadings.iloc[:, i].abs() > threshold
        if high_loadings.sum() > 1:
            redundant_features.append(loadings.index[high_loadings].tolist())
    
    redundant_features_list.append(redundant_features)
    
    print("PCA Loadings:")
    print(loadings)
    print("\nRedundant Features (High Correlation in the Same PC):")
    print(redundant_features)

# Plot all crops in a single figure (Explained Variance)
plot_pca_results(crops, explained_variances)

# Plot PCA scatter plots for clustering visualization
plot_pca_clusters_with_vectors(crops, pca_results, df, loadings_list, nutrients)
