"""Test configuration and shared fixtures"""
import os
import sys

# Add the project root directory to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)