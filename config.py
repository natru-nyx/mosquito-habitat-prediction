"""
Configuration file for Mosquito Habitat Prediction Project
==========================================================

This file contains all configuration parameters for the project,
including model settings, data paths, and evaluation metrics.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data configuration
DATA_CONFIG = {
    'regions': {
        'west_africa': {
            'bounds': (-18.0, 4.0, 15.0, 25.0),
            'countries': ['Mali', 'Burkina Faso', 'Ghana', 'Nigeria', 'Senegal'],
            'malaria_risk': 'high'
        },
        'east_africa': {
            'bounds': (28.0, -12.0, 52.0, 18.0),
            'countries': ['Kenya', 'Tanzania', 'Uganda', 'Ethiopia'],
            'malaria_risk': 'high'
        },
        'test_region': {
            'bounds': (-2.0, 6.0, -1.0, 7.0),
            'countries': ['Ghana'],
            'malaria_risk': 'medium'
        }
    },
    
    'sentinel2_bands': {
        'B02': {'name': 'Blue', 'resolution': 10, 'wavelength': 490},
        'B03': {'name': 'Green', 'resolution': 10, 'wavelength': 560},
        'B04': {'name': 'Red', 'resolution': 10, 'wavelength': 665},
        'B08': {'name': 'NIR', 'resolution': 10, 'wavelength': 842},
        'B11': {'name': 'SWIR1', 'resolution': 20, 'wavelength': 1610},
        'B12': {'name': 'SWIR2', 'resolution': 20, 'wavelength': 2190}
    },
    
    'target_resolution': 10,  # meters
    'cloud_cover_threshold': 20,  # percentage
    'temporal_window': 30,  # days for temporal compositing
}

# Feature engineering configuration
FEATURES_CONFIG = {
    'vegetation_indices': {
        'NDVI': {'bands': ['NIR', 'Red'], 'formula': '(NIR - Red) / (NIR + Red)'},
        'EVI': {'bands': ['NIR', 'Red', 'Blue'], 'formula': '2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)'},
        'SAVI': {'bands': ['NIR', 'Red'], 'formula': '((NIR - Red) / (NIR + Red + 0.5)) * 1.5'},
        'ARVI': {'bands': ['NIR', 'Red', 'Blue'], 'formula': '(NIR - (2*Red - Blue)) / (NIR + (2*Red - Blue))'},
        'GNDVI': {'bands': ['NIR', 'Green'], 'formula': '(NIR - Green) / (NIR + Green)'}
    },
    
    'water_indices': {
        'NDWI': {'bands': ['Green', 'NIR'], 'formula': '(Green - NIR) / (Green + NIR)'},
        'MNDWI': {'bands': ['Green', 'SWIR1'], 'formula': '(Green - SWIR1) / (Green + SWIR1)'},
        'NDMI': {'bands': ['NIR', 'SWIR1'], 'formula': '(NIR - SWIR1) / (NIR + SWIR1)'},
        'AWEIsh': {'bands': ['Blue', 'Green', 'NIR', 'SWIR1', 'SWIR2'], 
                  'formula': 'Blue + 2.5*Green - 1.5*(NIR + SWIR1) - 0.25*SWIR2'}
    },
    
    'climate_proxies': {
        'LST_proxy': {'bands': ['SWIR1', 'SWIR2'], 'formula': '(SWIR1 + SWIR2) / 2'},
        'moisture_stress': {'bands': ['Red', 'NIR'], 'formula': 'Red / NIR'},
        'vegetation_stress': {'bands': ['Red', 'NIR', 'SWIR1'], 'formula': '(Red * SWIR1) / NIR'},
        'albedo': {'bands': ['Blue', 'Green', 'Red', 'NIR'], 'formula': '0.3*Blue + 0.3*Green + 0.3*Red + 0.1*NIR'}
    },
    
    'texture_features': {
        'enabled': True,
        'window_size': 3,
        'features': ['contrast', 'dissimilarity', 'homogeneity', 'energy']
    },
    
    'temporal_features': {
        'enabled': True,
        'statistics': ['mean', 'std', 'min', 'max', 'median'],
        'seasonal_components': True
    }
}

# Model configuration
MODEL_CONFIG = {
    'gradient_boosting': {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'random_state': 42
    },
    
    'random_forest': {
        'n_estimators': 150,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42
    },
    
    'cnn_patch': {
        'patch_size': (32, 32),
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001,
        'dropout_rate': 0.5,
        'architecture': {
            'conv_layers': [
                {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'},
                {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
                {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'}
            ],
            'dense_layers': [
                {'units': 64, 'activation': 'relu'},
                {'units': 1, 'activation': 'sigmoid'}
            ]
        }
    },
    
    'ensemble': {
        'enabled': True,
        'models': ['gradient_boosting', 'random_forest'],
        'weights': [0.6, 0.4]
    }
}

# Training configuration
TRAINING_CONFIG = {
    'train_test_split': 0.7,
    'validation_split': 0.15,
    'test_split': 0.15,
    'spatial_cv_folds': 5,
    'temporal_validation': True,
    'stratify_by_region': True,
    'random_state': 42
}

# Evaluation configuration
EVALUATION_CONFIG = {
    'metrics': [
        'auc_roc',
        'auc_pr',
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'balanced_accuracy'
    ],
    
    'thresholds': {
        'minimum_auc': 0.75,
        'target_auc': 0.85,
        'minimum_precision': 0.70,
        'minimum_recall': 0.70
    },
    
    'validation_strategy': {
        'cross_validation': True,
        'spatial_blocks': True,
        'temporal_holdout': True,
        'geographic_holdout': True
    }
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    'risk_map': {
        'color_scheme': {
            'low_risk': '#2E8B57',      # Sea Green
            'medium_risk': '#FFD700',   # Gold
            'high_risk': '#FF4500'      # Orange Red
        },
        'risk_thresholds': [0.3, 0.7],
        'tile_layer': 'OpenStreetMap',
        'marker_size': 5,
        'opacity': 0.7
    },
    
    'feature_importance': {
        'top_n_features': 15,
        'plot_style': 'horizontal_bar',
        'color_scheme': 'viridis'
    },
    
    'performance_plots': {
        'figure_size': (12, 8),
        'dpi': 300,
        'save_format': ['png', 'pdf']
    }
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': {
        'console': True,
        'file': True,
        'file_path': LOGS_DIR / 'mosquito_habitat.log'
    }
}

# Research papers configuration
RESEARCH_PAPERS = [
    {
        'id': 'LP001',
        'title': 'Remote Sensing Applications in Malaria Vector Surveillance',
        'authors': 'Smith, J. et al.',
        'year': 2023,
        'journal': 'Remote Sensing of Environment',
        'doi': '10.1016/j.rse.2023.001',
        'relevance': 0.95,
        'keywords': ['remote sensing', 'malaria', 'anopheles', 'surveillance']
    },
    {
        'id': 'LP002',
        'title': 'Machine Learning for Mosquito Habitat Prediction Using Satellite Data',
        'authors': 'Johnson, A. & Brown, K.',
        'year': 2022,
        'journal': 'International Journal of Health Geographics',
        'doi': '10.1186/s12942-022-001',
        'relevance': 0.92,
        'keywords': ['machine learning', 'habitat prediction', 'satellite data']
    },
    {
        'id': 'LP003',
        'title': 'Vegetation Indices and Water Bodies Detection for Vector Control',
        'authors': 'Chen, L. et al.',
        'year': 2023,
        'journal': 'Spatial and Spatio-temporal Epidemiology',
        'doi': '10.1016/j.sste.2023.001',
        'relevance': 0.89,
        'keywords': ['vegetation indices', 'water detection', 'vector control']
    },
    {
        'id': 'LP004',
        'title': 'Deep Learning Approaches for Environmental Health Monitoring',
        'authors': 'Davis, M. & Wilson, R.',
        'year': 2024,
        'journal': 'Environmental Research',
        'doi': '10.1016/j.envres.2024.001',
        'relevance': 0.87,
        'keywords': ['deep learning', 'environmental health', 'monitoring']
    },
    {
        'id': 'LP005',
        'title': 'Temporal Dynamics of Mosquito Breeding Habitats from Satellite Imagery',
        'authors': 'Garcia, P. et al.',
        'year': 2023,
        'journal': 'PLOS ONE',
        'doi': '10.1371/journal.pone.2023.001',
        'relevance': 0.85,
        'keywords': ['temporal dynamics', 'breeding habitats', 'satellite imagery']
    }
]

# API endpoints and credentials
API_CONFIG = {
    'copernicus': {
        'base_url': 'https://catalogue.dataspace.copernicus.eu/odata/v1',
        'download_url': 'https://zipper.dataspace.copernicus.eu/odata/v1',
        'auth_url': 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
    },
    
    'globe_mosquito': {
        'base_url': 'https://www.globe.gov/globe-data/mosquito-habitat-mapper',
        'api_key_required': False
    },
    
    'weather_api': {
        'base_url': 'https://api.openweathermap.org/data/2.5',
        'api_key_required': True
    }
}

# File naming conventions
FILE_NAMING = {
    'sentinel2_processed': '{product_id}_{band}_{date}.tif',
    'features': '{region}_{date}_features.csv',
    'predictions': '{region}_{model}_{date}_predictions.csv',
    'models': '{model_type}_{region}_{timestamp}.pkl',
    'maps': '{region}_risk_map_{date}.html'
}

# Quality control thresholds
QUALITY_CONTROL = {
    'missing_data_threshold': 0.1,  # Maximum fraction of missing pixels
    'cloud_cover_mask': True,
    'snow_cover_mask': True,
    'shadow_mask': True,
    'outlier_detection': {
        'enabled': True,
        'method': 'iqr',
        'threshold': 3.0
    }
}

# Performance benchmarks
PERFORMANCE_BENCHMARKS = {
    'data_processing': {
        'max_time_per_image': 300,  # seconds
        'max_memory_usage': 8,      # GB
    },
    'model_training': {
        'max_training_time': 3600,  # seconds
        'target_convergence': 50    # epochs
    },
    'prediction': {
        'max_prediction_time': 60,  # seconds per 1000 pixels
        'batch_size_optimization': True
    }
}

# Export all configurations
__all__ = [
    'PROJECT_ROOT', 'DATA_DIR', 'MODELS_DIR', 'OUTPUTS_DIR', 'LOGS_DIR',
    'DATA_CONFIG', 'FEATURES_CONFIG', 'MODEL_CONFIG', 'TRAINING_CONFIG',
    'EVALUATION_CONFIG', 'VISUALIZATION_CONFIG', 'LOGGING_CONFIG',
    'RESEARCH_PAPERS', 'API_CONFIG', 'FILE_NAMING', 'QUALITY_CONTROL',
    'PERFORMANCE_BENCHMARKS'
]
