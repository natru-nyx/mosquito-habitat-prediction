"""
Mosquito Habitat Risk Prediction from Satellite Data
====================================================

A geospatial model to predict Anopheles-friendly habitats using Sentinel-2 imagery
and climate data for malaria ecology research.

Project Timeline: 12 weeks
- W1-3: Data acquisition and preprocessing
- W4-6: Feature engineering (NDVI/NDWI/LST proxies)
- W7-9: Model development (GBM/CNN-patch)
- W10-12: Evaluation and dashboard creation

Author: Research Team
Date: August 2025
"""

import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import folium
import warnings
warnings.filterwarnings('ignore')

class SentinelDataProcessor:
    """
    Class for processing Sentinel-2 satellite imagery data for mosquito habitat analysis.
    """
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.bands = {
            'B02': 'Blue',
            'B03': 'Green', 
            'B04': 'Red',
            'B08': 'NIR',
            'B11': 'SWIR1',
            'B12': 'SWIR2'
        }
        
    def calculate_vegetation_indices(self, nir, red, green, blue, swir1, swir2):
        """
        Calculate vegetation and water indices from Sentinel-2 bands.
        
        Parameters:
        -----------
        nir, red, green, blue, swir1, swir2 : numpy arrays
            Spectral bands from Sentinel-2
            
        Returns:
        --------
        dict: Dictionary containing calculated indices
        """
        
        # Avoid division by zero
        epsilon = 1e-10
        
        indices = {}
        
        # Normalized Difference Vegetation Index (NDVI)
        indices['NDVI'] = (nir - red) / (nir + red + epsilon)
        
        # Normalized Difference Water Index (NDWI)
        indices['NDWI'] = (green - nir) / (green + nir + epsilon)
        
        # Modified NDWI using SWIR
        indices['MNDWI'] = (green - swir1) / (green + swir1 + epsilon)
        
        # Enhanced Vegetation Index (EVI)
        indices['EVI'] = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + epsilon)
        
        # Soil Adjusted Vegetation Index (SAVI)
        L = 0.5  # soil brightness correction factor
        indices['SAVI'] = ((nir - red) / (nir + red + L)) * (1 + L)
        
        # Normalized Difference Moisture Index (NDMI)
        indices['NDMI'] = (nir - swir1) / (nir + swir1 + epsilon)
        
        return indices
    
    def extract_climate_proxies(self, bands_dict, date_info):
        """
        Extract climate-related features from satellite data.
        
        Parameters:
        -----------
        bands_dict : dict
            Dictionary containing spectral bands
        date_info : dict
            Date and time information
            
        Returns:
        --------
        dict: Climate proxy features
        """
        
        climate_features = {}
        
        # Land Surface Temperature proxy (using thermal characteristics)
        # Simplified approximation using SWIR bands
        swir1 = bands_dict.get('SWIR1', bands_dict.get('B11'))
        swir2 = bands_dict.get('SWIR2', bands_dict.get('B12'))
        
        if swir1 is not None and swir2 is not None:
            climate_features['LST_proxy'] = (swir1 + swir2) / 2
        
        # Moisture stress indicator
        nir = bands_dict.get('NIR', bands_dict.get('B08'))
        red = bands_dict.get('Red', bands_dict.get('B04'))
        
        if nir is not None and red is not None:
            climate_features['moisture_stress'] = red / (nir + 1e-10)
        
        # Seasonal indicators
        if 'month' in date_info:
            climate_features['season_sin'] = np.sin(2 * np.pi * date_info['month'] / 12)
            climate_features['season_cos'] = np.cos(2 * np.pi * date_info['month'] / 12)
        
        return climate_features

class MosquitoHabitatPredictor:
    """
    Machine learning model for predicting mosquito habitat suitability.
    """
    
    def __init__(self, model_type='gradient_boosting'):
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        
    def prepare_features(self, indices_dict, climate_dict, spatial_features=None):
        """
        Combine all features for model training.
        
        Parameters:
        -----------
        indices_dict : dict
            Vegetation and water indices
        climate_dict : dict
            Climate proxy features
        spatial_features : dict, optional
            Additional spatial features
            
        Returns:
        --------
        numpy.ndarray: Feature matrix
        """
        
        features = []
        feature_names = []
        
        # Add vegetation/water indices
        for name, values in indices_dict.items():
            if isinstance(values, np.ndarray):
                features.append(values.flatten())
                feature_names.append(name)
        
        # Add climate features
        for name, values in climate_dict.items():
            if isinstance(values, np.ndarray):
                features.append(values.flatten())
                feature_names.append(name)
        
        # Add spatial features if provided
        if spatial_features:
            for name, values in spatial_features.items():
                if isinstance(values, np.ndarray):
                    features.append(values.flatten())
                    feature_names.append(name)
        
        self.feature_names = feature_names
        
        if features:
            return np.column_stack(features)
        else:
            return np.array([])
    
    def train_gradient_boosting(self, X, y, **kwargs):
        """
        Train gradient boosting model for habitat prediction.
        """
        
        self.model = GradientBoostingClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 6),
            learning_rate=kwargs.get('learning_rate', 0.1),
            random_state=42
        )
        
        self.model.fit(X, y)
        self.feature_importance = self.model.feature_importances_
        
        return self.model
    
    def train_cnn_patch(self, X_patches, y, patch_size=(32, 32), **kwargs):
        """
        Train CNN model using image patches.
        
        Parameters:
        -----------
        X_patches : numpy.ndarray
            Image patches (n_samples, height, width, channels)
        y : numpy.ndarray
            Target labels
        patch_size : tuple
            Size of image patches
        """
        
        n_channels = X_patches.shape[-1] if len(X_patches.shape) == 4 else 1
        
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', 
                   input_shape=(*patch_size, n_channels)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        # Train the model
        history = model.fit(
            X_patches, y,
            epochs=kwargs.get('epochs', 50),
            batch_size=kwargs.get('batch_size', 32),
            validation_split=0.2,
            verbose=1
        )
        
        return model, history
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Returns:
        --------
        dict: Evaluation metrics
        """
        
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Make predictions
        if self.model_type == 'cnn':
            y_pred_proba = self.model.predict(X_test).flatten()
        else:
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        
        metrics = {
            'auc_score': auc_score,
            'precision': precision,
            'recall': recall,
            'classification_report': classification_report(y_test, y_pred)
        }
        
        return metrics

class HabitatRiskMapper:
    """
    Create risk maps and visualizations for mosquito habitat predictions.
    """
    
    def __init__(self):
        self.risk_map = None
        
    def create_risk_map(self, predictions, coordinates, region_bounds=None):
        """
        Create a risk map from model predictions.
        
        Parameters:
        -----------
        predictions : numpy.ndarray
            Model predictions (risk scores)
        coordinates : tuple
            (longitude, latitude) coordinates for each prediction
        region_bounds : tuple, optional
            (min_lon, min_lat, max_lon, max_lat) for map bounds
        """
        
        # Create folium map
        if region_bounds:
            center_lat = (region_bounds[1] + region_bounds[3]) / 2
            center_lon = (region_bounds[0] + region_bounds[2]) / 2
        else:
            center_lat = np.mean(coordinates[1])
            center_lon = np.mean(coordinates[0])
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='OpenStreetMap'
        )
        
        # Add risk points
        for i, (lon, lat) in enumerate(zip(coordinates[0], coordinates[1])):
            risk_score = predictions[i]
            
            # Color coding based on risk level
            if risk_score > 0.7:
                color = 'red'
                risk_level = 'High'
            elif risk_score > 0.4:
                color = 'orange'
                risk_level = 'Medium'
            else:
                color = 'green'
                risk_level = 'Low'
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                popup=f'Risk: {risk_level} ({risk_score:.3f})',
                color=color,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
        
        self.risk_map = m
        return m
    
    def plot_evaluation_metrics(self, metrics):
        """
        Plot model evaluation metrics.
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # AUC Score
        axes[0, 0].bar(['AUC Score'], [metrics['auc_score']], color='skyblue')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].set_title('Area Under ROC Curve')
        axes[0, 0].set_ylabel('AUC')
        
        # Precision-Recall Curve
        axes[0, 1].plot(metrics['recall'], metrics['precision'], linewidth=2)
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].grid(True)
        
        # Feature importance (if available)
        if hasattr(self, 'feature_importance') and self.feature_importance is not None:
            feature_names = getattr(self, 'feature_names', 
                                  [f'Feature_{i}' for i in range(len(self.feature_importance))])
            
            # Sort features by importance
            sorted_idx = np.argsort(self.feature_importance)[-10:]  # Top 10
            
            axes[1, 0].barh(np.array(feature_names)[sorted_idx], 
                           self.feature_importance[sorted_idx])
            axes[1, 0].set_title('Top 10 Feature Importance')
            axes[1, 0].set_xlabel('Importance')
        
        # Classification report visualization
        axes[1, 1].text(0.1, 0.5, metrics['classification_report'], 
                        fontfamily='monospace', fontsize=10,
                        verticalalignment='center')
        axes[1, 1].set_title('Classification Report')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig

def generate_sample_data(n_samples=1000, region='west_africa'):
    """
    Generate sample data for testing the mosquito habitat prediction pipeline.
    
    Parameters:
    -----------
    n_samples : int
        Number of sample points to generate
    region : str
        Geographic region for coordinate bounds
        
    Returns:
    --------
    dict: Sample dataset with features and labels
    """
    
    np.random.seed(42)
    
    # Define region bounds
    region_bounds = {
        'west_africa': (-18.0, 4.0, 15.0, 25.0),  # (min_lon, min_lat, max_lon, max_lat)
        'east_africa': (28.0, -12.0, 52.0, 18.0),
        'southeast_asia': (90.0, -10.0, 140.0, 28.0)
    }
    
    bounds = region_bounds.get(region, region_bounds['west_africa'])
    
    # Generate random coordinates within bounds
    lons = np.random.uniform(bounds[0], bounds[2], n_samples)
    lats = np.random.uniform(bounds[1], bounds[3], n_samples)
    
    # Generate synthetic spectral bands (simulate Sentinel-2 values)
    blue = np.random.uniform(0.05, 0.15, n_samples)
    green = np.random.uniform(0.05, 0.2, n_samples)
    red = np.random.uniform(0.04, 0.3, n_samples)
    nir = np.random.uniform(0.1, 0.6, n_samples)
    swir1 = np.random.uniform(0.05, 0.4, n_samples)
    swir2 = np.random.uniform(0.02, 0.3, n_samples)
    
    # Calculate indices
    processor = SentinelDataProcessor()
    indices = processor.calculate_vegetation_indices(nir, red, green, blue, swir1, swir2)
    
    # Generate climate features
    months = np.random.randint(1, 13, n_samples)
    date_info = {'month': months}
    bands_dict = {'NIR': nir, 'Red': red, 'SWIR1': swir1, 'SWIR2': swir2}
    climate = processor.extract_climate_proxies(bands_dict, date_info)
    
    # Generate labels based on environmental conditions
    # Higher probability near water bodies (high NDWI) and moderate vegetation (NDVI)
    prob_mosquito = (
        0.3 * (indices['NDWI'] > 0) +  # Water presence
        0.2 * ((indices['NDVI'] > 0.2) & (indices['NDVI'] < 0.6)) +  # Moderate vegetation
        0.2 * (climate['LST_proxy'] > np.median(climate['LST_proxy'])) +  # Warm areas
        0.1 * np.sin(2 * np.pi * months / 12)  # Seasonal effect
    )
    
    # Add noise and convert to binary labels
    prob_mosquito += np.random.normal(0, 0.1, n_samples)
    prob_mosquito = np.clip(prob_mosquito, 0, 1)
    labels = np.random.binomial(1, prob_mosquito)
    
    sample_data = {
        'coordinates': (lons, lats),
        'spectral_bands': {
            'blue': blue, 'green': green, 'red': red,
            'nir': nir, 'swir1': swir1, 'swir2': swir2
        },
        'indices': indices,
        'climate': climate,
        'labels': labels,
        'region_bounds': bounds
    }
    
    return sample_data

def main_pipeline():
    """
    Main pipeline for mosquito habitat risk prediction.
    """
    
    print("=== Mosquito Habitat Risk Prediction Pipeline ===")
    print("Generating sample data...")
    
    # Generate sample data
    data = generate_sample_data(n_samples=2000, region='west_africa')
    
    print(f"Generated {len(data['labels'])} samples")
    print(f"Positive samples (habitat present): {np.sum(data['labels'])}")
    print(f"Negative samples (habitat absent): {len(data['labels']) - np.sum(data['labels'])}")
    
    # Prepare features for machine learning
    predictor = MosquitoHabitatPredictor(model_type='gradient_boosting')
    
    # Combine all features
    feature_matrix = predictor.prepare_features(
        data['indices'], 
        data['climate']
    )
    
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Features: {predictor.feature_names}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, data['labels'], 
        test_size=0.3, random_state=42, stratify=data['labels']
    )
    
    print("\nTraining gradient boosting model...")
    # Train model
    model = predictor.train_gradient_boosting(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = predictor.evaluate_model(X_test, y_test)
    
    print(f"AUC Score: {metrics['auc_score']:.3f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    # Create risk map
    print("Creating risk map...")
    mapper = HabitatRiskMapper()
    
    # Make predictions for all data points
    predictions = model.predict_proba(feature_matrix)[:, 1]
    
    # Create interactive map
    risk_map = mapper.create_risk_map(
        predictions, 
        data['coordinates'], 
        data['region_bounds']
    )
    
    # Save map
    map_filename = 'mosquito_habitat_risk_map.html'
    risk_map.save(map_filename)
    print(f"Risk map saved as: {map_filename}")
    
    # Plot evaluation metrics
    print("Creating evaluation plots...")
    mapper.feature_importance = predictor.feature_importance
    mapper.feature_names = predictor.feature_names
    
    fig = mapper.plot_evaluation_metrics(metrics)
    plt.savefig('model_evaluation_metrics.png', dpi=300, bbox_inches='tight')
    print("Evaluation plots saved as: model_evaluation_metrics.png")
    
    # Feature importance analysis
    print("\nTop 5 Most Important Features:")
    feature_importance_df = pd.DataFrame({
        'Feature': predictor.feature_names,
        'Importance': predictor.feature_importance
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance_df.head())
    
    return {
        'model': model,
        'metrics': metrics,
        'predictions': predictions,
        'data': data,
        'feature_importance': feature_importance_df
    }

if __name__ == "__main__":
    # Run the main pipeline
    results = main_pipeline()
    
    print("\n=== Pipeline Complete ===")
    print("Generated outputs:")
    print("- mosquito_habitat_risk_map.html (Interactive risk map)")
    print("- model_evaluation_metrics.png (Model performance plots)")
    print("- Trained model with AUC score:", f"{results['metrics']['auc_score']:.3f}")
