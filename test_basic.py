#!/usr/bin/env python3
"""
Basic test script for AR-Agent
Tests core functionality without requiring full model downloads
"""

import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} available")
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available, will use CPU")
    except ImportError:
        print("✗ PyTorch not available")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__} available")
    except ImportError:
        print("✗ OpenCV not available")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} available")
    except ImportError:
        print("✗ NumPy not available")
        return False
    
    try:
        from flask import Flask
        print("✓ Flask available")
    except ImportError:
        print("✗ Flask not available")
        return False
    
    try:
        import yaml
        print("✓ PyYAML available")
    except ImportError:
        print("✗ PyYAML not available")
        return False
    
    return True

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    config_path = Path(__file__).parent / "configs" / "config.yaml"
    if not config_path.exists():
        print("✗ Config file not found")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Configuration loaded successfully")
        print(f"  Model: {config.get('model', {}).get('name', 'Unknown')}")
        print(f"  Device: {config.get('model', {}).get('device', 'Unknown')}")
        return True
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False

def test_directory_structure():
    """Test if all required directories exist"""
    print("\nTesting directory structure...")
    
    base_path = Path(__file__).parent
    required_dirs = [
        "src/medical_analyzer",
        "src/ar_interface",
        "templates",
        "static/css",
        "static/js",
        "configs",
        "data",
        "models",
        "cache",
        "uploads",
        "logs"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} missing")
            all_exist = False
    
    return all_exist

def test_flask_app():
    """Test if Flask app can be created"""
    print("\nTesting Flask app creation...")
    
    try:
        # Import app without running it
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Mock the model loading to avoid downloading
        import unittest.mock as mock
        
        with mock.patch('transformers.LlavaNextProcessor.from_pretrained'):
            with mock.patch('transformers.LlavaNextForConditionalGeneration.from_pretrained'):
                from app import app
                
        print("✓ Flask app created successfully")
        
        # Test basic routes
        with app.test_client() as client:
            response = client.get('/')
            if response.status_code == 200:
                print("✓ Index route working")
            else:
                print(f"✗ Index route failed: {response.status_code}")
                return False
            
            response = client.get('/health')
            if response.status_code == 200:
                print("✓ Health check route working")
            else:
                print(f"✗ Health check route failed: {response.status_code}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Flask app creation failed: {e}")
        return False

def test_medical_analyzer():
    """Test medical analyzer module"""
    print("\nTesting medical analyzer...")
    
    try:
        from medical_analyzer.analyzer import MedicalImageAnalyzer
        print("✓ MedicalImageAnalyzer imported successfully")
        
        # Test initialization without model loading
        import unittest.mock as mock
        with mock.patch('transformers.LlavaNextProcessor.from_pretrained'):
            with mock.patch('transformers.LlavaNextForConditionalGeneration.from_pretrained'):
                analyzer = MedicalImageAnalyzer()
                print("✓ MedicalImageAnalyzer initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Medical analyzer test failed: {e}")
        return False

def test_ar_engine():
    """Test AR engine module"""
    print("\nTesting AR engine...")
    
    try:
        from ar_interface.ar_engine import AREngine, ARMode
        print("✓ AREngine imported successfully")
        
        # Test enum
        modes = list(ARMode)
        print(f"✓ AR modes available: {[mode.value for mode in modes]}")
        
        return True
        
    except Exception as e:
        print(f"✗ AR engine test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("AR-Agent Basic Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_config),
        ("Directory Structure Test", test_directory_structure),
        ("Flask App Test", test_flask_app),
        ("Medical Analyzer Test", test_medical_analyzer),
        ("AR Engine Test", test_ar_engine)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! AR-Agent is ready to use.")
        return 0
    else:
        print("⚠ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())