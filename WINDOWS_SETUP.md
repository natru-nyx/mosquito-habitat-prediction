# Quick Setup Guide for Windows

## Step 1: Install Python
1. Go to https://www.python.org/downloads/windows/
2. Download Python 3.10+ (latest stable version)
3. **IMPORTANT**: During installation, check "Add Python to PATH"
4. Verify installation: Open new PowerShell and run `python --version`

## Step 2: Install Git
1. Go to https://git-scm.com/download/win
2. Download and install Git for Windows
3. Use default settings during installation
4. Verify installation: Open new PowerShell and run `git --version`

## Step 3: Set Up Project
Open PowerShell in your project folder and run:

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt

# Create missing directories
mkdir models
mkdir outputs  
mkdir logs

# Initialize Git repository
git init
git add .
git commit -m "Initial commit"

# Run the dashboard
python web_dashboard.py
```

## Step 4: Access Dashboard
- Open your browser
- Go to: http://localhost:8080
- You should see the Mosquito Habitat Risk Prediction dashboard

## Alternative Entry Points
If web_dashboard.py doesn't work, try:
- `python gui_dashboard.py` (GUI version)
- `python mosquito_habitat_prediction.py` (analysis pipeline)

## Troubleshooting
- If Python commands fail: Restart PowerShell after Python installation
- If Git commands fail: Restart PowerShell after Git installation
- If dependencies fail to install: Try `pip install --upgrade pip` first