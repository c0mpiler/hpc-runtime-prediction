#!/usr/bin/env python3
"""
Analytics Page - Display model performance metrics and system analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Import utilities
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.grpc_client import format_runtime

def create_gauge_chart(value: float, title: str, max_value: float = 100, 
                      suffix: str = "%", threshold_good: float = 80, 
                      threshold_warning: float = 60) -> go.Figure:
    """Create a gauge chart for metrics."""
    
    # Determine color based on thresholds
    if value >= threshold_good:
        color = "green"
    elif value >= threshold_warning:
        color = "orange"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        number={'suffix': suffix},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, threshold_warning], 'color': "lightgray"},
                {'range': [threshold_warning, threshold_good], 'color': "lightyellow"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold_good
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def show_analytics(client):
    """Display analytics dashboard."""
    
    st.header("📊 Analytics Dashboard")
    
    # Check API health
    health_status = client.health_check()
    
    if not health_status.get('serving', False):
        st.error("❌ API service is not available. Cannot fetch analytics.")
        st.info(f"Status: {health_status.get('message', 'Unknown error')}")
        return
    
    # Get model info
    try:
        from proto import rt_predictor_pb2
        request = rt_predictor_pb2.GetModelInfoRequest()
        response = client.stub.GetModelInfo(request)
        model_info = {
            'version': response.model_version,
            'type': response.model_type,
            'training_date': response.training_date,
            'num_features': response.num_features,
            'training_samples': response.training_samples,
            'validation_mae': response.validation_mae,
            'validation_mape': response.validation_mape,
            'predictions_served': response.predictions_served,
            'average_latency_ms': response.average_latency_ms,
            'uptime_seconds': response.uptime_seconds,
            'ensemble_weights': dict(response.ensemble_weights)
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        model_info = None
    
    # System Health Section
    st.subheader("🏥 System Health")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_color = "green" if health_status.get('serving') else "red"
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background-color: #{status_color}22; 
                    border-radius: 10px; border: 2px solid #{status_color}44;'>
            <h3 style='margin: 0; color: {status_color};'>API Status</h3>
            <h2 style='margin: 10px 0;'>{"🟢 Online" if health_status.get('serving') else "🔴 Offline"}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        uptime_hours = health_status.get('uptime_seconds', 0) / 3600
        st.metric(
            "Uptime",
            f"{uptime_hours:.1f} hours",
            delta=f"{health_status.get('uptime_seconds', 0)} seconds"
        )
    
    with col3:
        cpu_usage = health_status.get('cpu_usage_percent', 0)
        st.metric(
            "CPU Usage",
            f"{cpu_usage:.1f}%",
            delta=f"{cpu_usage - 50:.1f}%" if cpu_usage > 0 else None
        )
    
    with col4:
        memory_usage = health_status.get('memory_usage_mb', 0)
        st.metric(
            "Memory Usage",
            f"{memory_usage:.1f} MB",
            delta=None
        )
    
    # Model Performance Section
    if model_info:
        st.subheader("🎯 Model Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Model accuracy gauge
            if model_info.get('validation_mape', 0) > 0:
                accuracy = 100 - model_info['validation_mape']
                fig = create_gauge_chart(
                    accuracy,
                    "Model Accuracy",
                    max_value=100,
                    threshold_good=90,
                    threshold_warning=80
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average latency gauge
            avg_latency = model_info.get('average_latency_ms', 0)
            fig = create_gauge_chart(
                avg_latency,
                "Avg Latency",
                max_value=100,
                suffix=" ms",
                threshold_good=20,
                threshold_warning=50
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Predictions served
            st.metric(
                "Total Predictions",
                f"{model_info.get('predictions_served', 0):,}",
                delta=None
            )
            st.metric(
                "Model Version",
                model_info.get('version', 'Unknown'),
                delta=None
            )
    
    # Model Details Section
    if model_info:
        st.subheader("🔬 Model Details")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Model information table
            model_details = {
                "Property": ["Model Type", "Version", "Training Date", "Features", 
                           "Training Samples", "Validation MAE", "Validation MAPE"],
                "Value": [
                    model_info.get('type', 'Unknown'),
                    model_info.get('version', 'Unknown'),
                    model_info.get('training_date', 'Unknown'),
                    str(model_info.get('num_features', 0)),
                    f"{model_info.get('training_samples', 0):,}",
                    f"{model_info.get('validation_mae', 0):.2f} seconds",
                    f"{model_info.get('validation_mape', 0):.2f}%"
                ]
            }
            
            df_details = pd.DataFrame(model_details)
            st.dataframe(df_details, hide_index=True, use_container_width=True)
        
        with col2:
            # Ensemble weights pie chart
            if model_info.get('ensemble_weights'):
                weights = model_info['ensemble_weights']
                fig = px.pie(
                    values=list(weights.values()),
                    names=list(weights.keys()),
                    title="Ensemble Model Weights"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Live Metrics Section (Simulated)
    st.subheader("📈 Live Metrics")
    
    # Create tabs for different metric views
    tab1, tab2, tab3 = st.tabs(["Request Rate", "Latency Distribution", "Error Rate"])
    
    with tab1:
        # Simulated request rate over time
        time_points = pd.date_range(end=datetime.now(), periods=60, freq='1min')
        request_rates = np.random.poisson(50, 60) + np.sin(np.linspace(0, 2*np.pi, 60)) * 10
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_points,
            y=request_rates,
            mode='lines',
            name='Requests/min',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title="Request Rate (Last Hour)",
            xaxis_title="Time",
            yaxis_title="Requests per Minute",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Simulated latency distribution
        latencies = np.random.gamma(2, 5, 1000)
        
        fig = px.histogram(
            latencies,
            nbins=50,
            title="Prediction Latency Distribution",
            labels={'value': 'Latency (ms)', 'count': 'Frequency'}
        )
        
        # Add percentile lines
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        fig.add_vline(x=p50, line_dash="dash", line_color="green", 
                     annotation_text=f"p50: {p50:.1f}ms")
        fig.add_vline(x=p95, line_dash="dash", line_color="orange", 
                     annotation_text=f"p95: {p95:.1f}ms")
        fig.add_vline(x=p99, line_dash="dash", line_color="red", 
                     annotation_text=f"p99: {p99:.1f}ms")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Simulated error rate
        time_points = pd.date_range(end=datetime.now(), periods=24, freq='1H')
        error_rates = np.random.exponential(0.5, 24) * np.random.choice([0, 1], 24, p=[0.7, 0.3])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=time_points,
            y=error_rates,
            name='Error Rate',
            marker_color=['red' if x > 1 else 'green' for x in error_rates]
        ))
        
        fig.update_layout(
            title="Error Rate (Last 24 Hours)",
            xaxis_title="Time",
            yaxis_title="Error Rate (%)",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # System Recommendations
    st.subheader("💡 Recommendations")
    
    recommendations = []
    
    if health_status.get('cpu_usage_percent', 0) > 80:
        recommendations.append(("⚠️ High CPU Usage", 
                              "Consider scaling the API service or optimizing model inference"))
    
    if health_status.get('memory_usage_mb', 0) > 4000:
        recommendations.append(("⚠️ High Memory Usage", 
                              "Monitor for memory leaks or consider increasing resources"))
    
    if model_info and model_info.get('average_latency_ms', 0) > 50:
        recommendations.append(("⚠️ High Latency", 
                              "Consider enabling model caching or optimizing feature engineering"))
    
    if not recommendations:
        recommendations.append(("✅ System Healthy", 
                              "All metrics are within normal ranges"))
    
    for title, desc in recommendations:
        st.info(f"**{title}**: {desc}")
    
    # Refresh button
    col1, col2, col3 = st.columns([1, 1, 3])
    with col2:
        if st.button("🔄 Refresh Analytics", type="primary"):
            st.rerun()
