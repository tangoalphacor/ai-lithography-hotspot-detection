"""
Demo script to test the About page functionality
"""
import streamlit as st
import sys
import os

# Add the project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# Import the about page
from pages.about import show_about_page

# Configure the page
st.set_page_config(
    page_title="About - AI Lithography Hotspot Detection",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Show the about page
show_about_page()
