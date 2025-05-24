#!/usr/bin/env python3
"""
RT Predictor Web UI - Streamlit interface for job runtime prediction.
Connects to the RT Predictor API service via gRPC.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from pathlib import Path
import logging
import os
from typing import Optional, Dict, List, Tuple
import sys

# Page configuration
st.set_page_config(
    page_title="RT Predictor - HPC Job Runtime Prediction",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import gRPC client
try:
    from utils.grpc_client import RTPredictorClient, PredictionResult, format_runtime, calculate_efficiency
except ImportError as e:
    st.error(f"Failed to import gRPC client: {str(e)}")
    st.stop()

# Environment variables
API_HOST = os.getenv('RT_PREDICTOR_API_HOST', 'localhost')
API_PORT = int(os.getenv('RT_PREDICTOR_API_PORT', '50051'))

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Metrics styling */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.15);
    }
    
    /* Chart title styling */
    .chart-title {
        font-size: 20px !important;
        font-weight: 600;
        margin-top: 60px !important;
        margin-bottom: 40px !important;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'client' not in st.session_state:
    st.session_state.client = None
if 'api_connected' not in st.session_state:
    st.session_state.api_connected = False

# Function to connect to API
def connect_to_api():
    """Establish connection to the RT Predictor API service."""
    try:
        if st.session_state.client:
            st.session_state.client.close()
        
        st.session_state.client = RTPredictorClient(host=API_HOST, port=API_PORT)
        # If we got here, connection was successful
        st.session_state.api_connected = True
        logger.info(f"Connected to API service at {API_HOST}:{API_PORT}")
        return True
            
    except Exception as e:
        st.session_state.api_connected = False
        logger.error(f"Failed to connect to API: {str(e)}")
        return False

# Header
st.markdown("""
<div class="main-header">
    <h1>🚀 RT Predictor</h1>
    <p>Advanced Machine Learning for HPC Job Runtime Prediction</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    
    # API Connection Status
    st.subheader("API Connection")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.session_state.api_connected:
            st.success("✅ Connected to API")
        else:
            st.error("❌ Not connected")
    
    with col2:
        if st.button("🔄", help="Reconnect to API"):
            with st.spinner("Connecting..."):
                if connect_to_api():
                    st.success("Connected!")
                else:
                    st.error("Connection failed!")
    
    # Navigation
    st.subheader("🗺️ Navigation")
    page = st.radio(
        "Select Page",
        ["🎯 Single Prediction", "📦 Batch Prediction", "📊 Analytics"],
        index=0,
        label_visibility="collapsed"
    )

# Main content area
if not st.session_state.api_connected:
    # Try to connect on first load
    connect_to_api()

if st.session_state.api_connected and st.session_state.client:
    # Import pages
    from pages.single_prediction import show_single_prediction
    from pages.batch_prediction import show_batch_prediction
    from pages.analytics import show_analytics
    
    # Route to selected page
    if page == "🎯 Single Prediction":
        show_single_prediction(st.session_state.client)
    elif page == "📦 Batch Prediction":
        show_batch_prediction(st.session_state.client)
    elif page == "📊 Analytics":
        show_analytics(st.session_state.client)
else:
    # Connection error state
    st.error("⚠️ Unable to connect to RT Predictor API")
    st.markdown("""
    ### Troubleshooting Steps:
    
    1. **Check if the API service is running:**
       ```bash
       docker ps | grep rt-predictor-api
       ```
    
    2. **Verify the API is accessible:**
       ```bash
       grpcurl -plaintext localhost:50051 list
       ```
    
    3. **Check the logs:**
       ```bash
       docker logs rt-predictor-api
       ```
    
    4. **Restart the services:**
       ```bash
       docker-compose restart
       ```
    
    5. Click the 🔄 button in the sidebar to retry the connection.
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        RT Predictor v1.0 | Powered by XGBoost, LightGBM & CatBoost | 
        <a href='https://github.com/your-org/rt-predictor' target='_blank'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
