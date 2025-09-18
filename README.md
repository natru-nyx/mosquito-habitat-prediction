# Mosquito Habitat Risk Prediction from Satellite Data

## Project Overview

**Objective**: Build a geospatial model that predicts Anopheles-friendly habitats using Sentinel-2 imagery and climate features to support malaria prevention efforts.

**Duration**: 12 weeks

**Novel Approach**: Compare classic indices (NDVI/NDWI) vs learned pixel embeddings for habitat prediction.

## Project Timeline

### Weeks 1-3: Data Acquisition & Preprocessing
- [ ] Set up Copernicus Data Space account
- [ ] Download Sentinel-2 imagery for target regions
- [ ] Implement data preprocessing pipeline
- [ ] Create training/validation datasets

### Weeks 4-6: Feature Engineering
- [ ] Calculate vegetation indices (NDVI, EVI, SAVI)
- [ ] Compute water indices (NDWI, MNDWI)
- [ ] Extract Land Surface Temperature (LST) proxies
- [ ] Develop climate feature engineering pipeline

### Weeks 7-9: Model Development
- [ ] Implement Gradient Boosting Machine (GBM)
- [ ] Develop CNN patch-based model
- [ ] Compare classic vs learned embeddings
- [ ] Hyperparameter optimization

### Weeks 10-12: Evaluation & Visualization
- [ ] Model validation with ground truth data
- [ ] Create interactive risk maps
- [ ] Develop dashboard for stakeholders
- [ ] Write technical paper

## Data Sources

### Primary Data
- **Sentinel-2 Imagery**: Free via Copernicus Data Space Ecosystem
  - URL: https://dataspace.copernicus.eu/
  - Bands: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR1), B12 (SWIR2)
  - Resolution: 10m-20m
  - Temporal: Every 5 days

### Validation Data
- **GLOBE Mosquito Habitat Mapper**: Citizen science data
  - URL: https://www.globe.gov/globe-data/globe-research/mosquito-habitat-mapper
  - Ground truth habitat observations
  - Global coverage with variable density

### Climate Data
- **ERA5 Reanalysis**: Temperature, precipitation, humidity
- **MODIS LST**: Land Surface Temperature validation
- **CHIRPS**: Precipitation data for tropical regions

## Technical Implementation

### Feature Engineering

#### Vegetation Indices
```python
# Normalized Difference Vegetation Index
NDVI = (NIR - Red) / (NIR + Red)

# Enhanced Vegetation Index
EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)

# Soil Adjusted Vegetation Index
SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)  # L = 0.5
```

#### Water Indices
```python
# Normalized Difference Water Index
NDWI = (Green - NIR) / (Green + NIR)

# Modified NDWI (using SWIR)
MNDWI = (Green - SWIR1) / (Green + SWIR1)

# Normalized Difference Moisture Index
NDMI = (NIR - SWIR1) / (NIR + SWIR1)
```

#### Climate Proxies
```python
# Land Surface Temperature proxy
LST_proxy = (SWIR1 + SWIR2) / 2

# Moisture stress indicator
moisture_stress = Red / NIR

# Seasonal indicators
season_sin = sin(2π * month / 12)
season_cos = cos(2π * month / 12)
```

### Model Architecture

#### Gradient Boosting Machine
- **Algorithm**: XGBoost/LightGBM
- **Features**: All indices + climate proxies
- **Optimization**: Bayesian optimization
- **Validation**: Spatial cross-validation

#### CNN Patch Model
```python
# Architecture
Conv2D(32, 3x3) → ReLU → MaxPool
Conv2D(64, 3x3) → ReLU → MaxPool  
Conv2D(64, 3x3) → ReLU
Flatten → Dense(64) → Dropout → Dense(1, sigmoid)

# Input: 32x32x6 patches (6 spectral bands)
# Output: Habitat probability [0,1]
```

## Evaluation Metrics

### Model Performance
- **AUC-ROC**: Area under ROC curve
- **PR-AUC**: Precision-Recall curve area
- **Spatial CV**: Account for spatial autocorrelation
- **Feature Importance**: SHAP values

### Validation Strategy
- **Hold-out regions**: Geographic separation
- **Temporal validation**: Train on past, test on recent
- **Cross-validation**: Spatial blocks to avoid data leakage

## Expected Outputs

### 1. Risk Maps
- **Format**: Interactive HTML maps (Folium)
- **Resolution**: 10m pixel resolution
- **Coverage**: Regional (country/state level)
- **Updates**: Monthly/seasonal

### 2. Model Performance
- **Accuracy Metrics**: AUC > 0.80 target
- **Feature Analysis**: Most predictive indices
- **Comparison**: Classic vs learned features

### 3. Code & Documentation
- **GitHub Repository**: Open source pipeline
- **Documentation**: API reference, tutorials
- **Reproducibility**: Docker containers, environment files

### 4. Research Paper
- **Target Journals**: 
  - Remote Sensing of Environment
  - International Journal of Health Geographics
  - Spatial and Spatio-temporal Epidemiology
- **Focus**: Methodological advancement + public health application

## Impact & Applications

### Public Health Benefits
- **Early Warning**: Identify high-risk areas before peak season
- **Resource Allocation**: Target interventions efficiently  
- **Cost Reduction**: Reduce unnecessary spraying/treatments
- **Scalability**: Apply to any region with satellite coverage

### NGO & Government Use Cases
- **WHO**: Support malaria elimination programs
- **Local Health Departments**: Operational planning
- **Research Organizations**: Epidemiological studies
- **Aid Organizations**: Resource deployment

## Technical Requirements

### Computational Resources
- **GPU**: For CNN training (8GB+ VRAM recommended)
- **Storage**: 500GB+ for imagery and processed data
- **RAM**: 16GB+ for large image processing
- **CPU**: Multi-core for parallel processing

### Software Stack
- **Python 3.8+**: Core programming language
- **GDAL/Rasterio**: Geospatial data processing
- **TensorFlow/PyTorch**: Deep learning
- **Scikit-learn**: Traditional ML algorithms
- **Folium/Plotly**: Interactive visualizations

## Risk Mitigation

### Data Availability
- **Backup Sources**: Multiple satellite platforms (Landsat, MODIS)
- **Cloud Coverage**: Multi-temporal compositing
- **Regional Gaps**: Focus on well-covered areas initially

### Model Performance
- **Baseline Models**: Simple logistic regression
- **Ensemble Methods**: Combine multiple approaches
- **Domain Expertise**: Collaborate with epidemiologists

### Validation Challenges
- **Ground Truth**: Use multiple validation sources
- **Spatial Bias**: Account for sampling patterns
- **Temporal Drift**: Regular model updates

## Success Criteria

### Technical Success
- [ ] AUC-ROC > 0.80 on held-out test regions
- [ ] Deployment-ready pipeline with <5min processing time
- [ ] Interactive dashboard accessible to non-technical users

### Research Success
- [ ] Publication in peer-reviewed journal
- [ ] Open-source code with >50 GitHub stars
- [ ] Presentation at international conference

### Impact Success
- [ ] Adoption by at least one NGO/government agency
- [ ] Integration into existing malaria surveillance systems
- [ ] Demonstrable improvement in intervention targeting

## Getting Started

### 1. Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd mosquito-habitat-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Acquisition
```bash
# Run data download script
python scripts/download_sentinel2.py --region west_africa --start_date 2023-01-01 --end_date 2023-12-31

# Process imagery
python scripts/preprocess_imagery.py --input_dir data/raw --output_dir data/processed
```

### 3. Model Training
```bash
# Train gradient boosting model
python train_gbm.py --config configs/gbm_config.yaml

# Train CNN model  
python train_cnn.py --config configs/cnn_config.yaml
```

### 4. Generate Risk Maps
```bash
# Create risk predictions
python predict.py --model_path models/best_model.pkl --region west_africa --output_dir outputs/

# Generate interactive map
python create_map.py --predictions outputs/predictions.csv --output mosquito_risk_map.html
```

## Contact & Collaboration

For questions, collaborations, or technical support:
- **Email**: [your-email@institution.edu]
- **GitHub**: [github.com/your-username/mosquito-habitat-prediction]
- **Twitter**: [@your_handle]

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Copernicus Programme for free satellite data access
- GLOBE Program for citizen science validation data
- WHO and malaria research community for domain expertise
- Open source scientific Python community
