# Mosquito Habitat Risk Prediction - Setup Instructions

## Current Status & Issues Found

### ❌ **Critical Issues Preventing Execution:**

1. **Python Not Available**
   - Python is not installed or not in system PATH
   - Commands `python` and `py` both fail
   - Cannot run any Python scripts without fixing this first

2. **Git Not Available**
   - Git is not installed or not in system PATH
   - Cannot initialize repository or version control

### 📋 **Project Inconsistencies Identified:**

#### Missing Files Referenced in Documentation:
- `predict.py` - Referenced in README.md but doesn't exist
- `create_map.py` - Referenced in README.md but doesn't exist
- `train_gbm.py` - Referenced in README.md but doesn't exist
- `train_cnn.py` - Referenced in README.md but doesn't exist
- `scripts/preprocess_imagery.py` - Referenced in README.md but doesn't exist

#### Directory Structure Issues:
- `models/` directory - Referenced in config.py but doesn't exist
- `outputs/` directory - Referenced in config.py but doesn't exist
- `logs/` directory - Referenced in config.py but doesn't exist

#### Platform Compatibility:
- `quick_start.sh` is a Bash script but you're on Windows with PowerShell
- Need Windows equivalent commands

#### Duplicate Files:
- `web_dashboard.py` and `web_dashboard_backup.py` (same functionality)
- `dashboard.py` vs `web_dashboard.py` vs `gui_dashboard.py` (unclear which is main)

### ✅ **What's Working:**
- All Python files have valid syntax (no syntax errors)
- Configuration file (`config.py`) is comprehensive and well-structured
- Documentation is detailed and professional
- Sample data exists in `data/` directory
- Requirements.txt lists all necessary dependencies

### 🔧 **Required Setup Steps:**

#### 1. Install Python
```powershell
# Download from https://www.python.org/downloads/windows/
# During installation, check "Add Python to PATH"
# Verify with: python --version
```

#### 2. Install Git
```powershell
# Download from https://git-scm.com/download/win
# Install with default settings
# Verify with: git --version
```

#### 3. Set up Python Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

#### 4. Create Missing Directories
```powershell
mkdir models, outputs, logs
```

#### 5. Initialize Git Repository
```powershell
git init
git add .
git commit -m "Initial commit"
```

#### 6. Run the Dashboard
```powershell
python web_dashboard.py
# Then open http://localhost:8080 in browser
```

### 📊 **Project Analysis Summary:**
- **Main Entry Points**: `web_dashboard.py` (recommended), `gui_dashboard.py`, or `mosquito_habitat_prediction.py`
- **Dependencies**: 15+ packages including TensorFlow, scikit-learn, rasterio, folium
- **Data Processing**: Handles Sentinel-2 satellite imagery and geospatial analysis
- **Output**: Interactive HTML maps, PNG visualizations, model metrics

The project appears to be a sophisticated satellite data analysis system for predicting mosquito habitats, but requires proper Python and Git installation before it can run.