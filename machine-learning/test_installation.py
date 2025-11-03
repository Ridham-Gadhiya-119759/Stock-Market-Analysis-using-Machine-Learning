"""
Quick test script to verify installation
"""
import sys
import os

def test_imports():
    """Test all module imports"""
    print("🔍 Testing imports...")
    
    try:
        import numpy as np
        print(f"✅ numpy {np.__version__}")
    except:
        print("❌ numpy failed")
        return False
    
    try:
        import pandas as pd
        print(f"✅ pandas {pd.__version__}")
    except:
        print("❌ pandas failed")
        return False
    
    try:
        import sklearn
        print(f"✅ scikit-learn {sklearn.__version__}")
    except:
        print("❌ scikit-learn failed")
        return False
    
    try:
        import matplotlib
        print(f"✅ matplotlib {matplotlib.__version__}")
    except:
        print("❌ matplotlib failed")
        return False
    
    try:
        import seaborn as sns
        print(f"✅ seaborn {sns.__version__}")
    except:
        print("❌ seaborn failed")
        return False
    
    try:
        import yfinance as yf
        print(f"✅ yfinance {yf.__version__}")
    except:
        print("❌ yfinance failed")
        return False
    
    try:
        import joblib
        print(f"✅ joblib {joblib.__version__}")
    except:
        print("❌ joblib failed")
        return False
    
    return True

def test_modules():
    """Test project modules"""
    print("\n🔍 Testing project modules...")
    
    try:
        from config import settings
        print("✅ config module")
    except Exception as e:
        print(f"❌ config module: {e}")
        return False
    
    try:
        from utils import logger
        print("✅ utils module")
    except Exception as e:
        print(f"❌ utils module: {e}")
        return False
    
    try:
        from data import DataLoader
        print("✅ data module")
    except Exception as e:
        print(f"❌ data module: {e}")
        return False
    
    try:
        from features import FeatureBuilder
        print("✅ features module")
    except Exception as e:
        print(f"❌ features module: {e}")
        return False
    
    try:
        from model import ModelTrainer
        print("✅ model module")
    except Exception as e:
        print(f"❌ model module: {e}")
        return False
    
    try:
        from evaluation import MetricsCalculator
        print("✅ evaluation module")
    except Exception as e:
        print(f"❌ evaluation module: {e}")
        return False
    
    return True

def test_directory_structure():
    """Test directory structure"""
    print("\n🔍 Testing directory structure...")
    
    required_dirs = [
        'config',
        'data',
        'features',
        'model',
        'evaluation',
        'utils',
        'outputs'
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/ exists")
        else:
            print(f"❌ {dir_name}/ missing")
            all_exist = False
    
    return all_exist

if __name__ == "__main__":
    print("="*70)
    print("🧪 STOCK FORECASTING SYSTEM - INSTALLATION TEST")
    print("="*70)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test modules
    modules_ok = test_modules()
    
    # Test directories
    dirs_ok = test_directory_structure()
    
    # Final result
    print("\n" + "="*70)
    if imports_ok and modules_ok and dirs_ok:
        print("✅ ALL TESTS PASSED! System is ready to use.")
        print("\n💡 Run 'python main.py' to start forecasting!")
    else:
        print("❌ SOME TESTS FAILED. Please check the errors above.")
        print("\n💡 Try running: pip install -r requirements.txt")
    print("="*70)
