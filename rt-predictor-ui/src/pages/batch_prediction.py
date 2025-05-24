#!/usr/bin/env python3
"""
Batch Prediction Page - Handle multiple job predictions via CSV upload.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import io
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Import utilities
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.grpc_client import format_runtime, calculate_efficiency

def validate_csv_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate CSV data for required columns and data types.
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    required_columns = {
        'cpus_req': 'int',
        'nodes_req': 'int',
        'mem_req': 'float',
        'walltime_req': 'float',
        'partition': 'str',
        'qos': 'str'
    }
    
    # Check for required columns
    missing_cols = set(required_columns.keys()) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Validate data types and values
    for col, dtype in required_columns.items():
        if col in df.columns:
            try:
                if dtype == 'int':
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                    if (df[col] <= 0).any():
                        errors.append(f"Column '{col}' must contain positive integers")
                elif dtype == 'float':
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                    if (df[col] <= 0).any():
                        errors.append(f"Column '{col}' must contain positive numbers")
                elif dtype == 'str':
                    df[col] = df[col].astype(str)
                    if df[col].str.strip().eq('').any():
                        errors.append(f"Column '{col}' cannot contain empty values")
            except Exception as e:
                errors.append(f"Error processing column '{col}': {str(e)}")
    
    return len(errors) == 0, errors

def show_batch_prediction(client):
    """Display batch prediction interface."""
    
    st.header("📦 Batch Prediction")
    st.markdown("""
    Upload a CSV file containing job parameters to get runtime predictions for multiple jobs at once.
    
    ### Required CSV Format:
    | cpus_req | nodes_req | mem_req | walltime_req | partition | qos |
    |----------|-----------|---------|--------------|-----------|-----|
    | 4        | 1         | 16.0    | 3600         | standard  | normal |
    | 8        | 2         | 32.0    | 7200         | gpu       | high |
    """)
    
    # File upload section
    st.subheader("📤 Upload CSV File")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with job parameters"
    )
    
    # Sample CSV download
    col1, col2 = st.columns([1, 4])
    with col1:
        sample_data = pd.DataFrame({
            'cpus_req': [4, 8, 16, 32],
            'nodes_req': [1, 2, 4, 8],
            'mem_req': [16.0, 32.0, 64.0, 128.0],
            'walltime_req': [3600, 7200, 14400, 28800],
            'partition': ['standard', 'gpu', 'standard', 'bigmem'],
            'qos': ['normal', 'high', 'normal', 'high']
        })
        
        csv_buffer = io.StringIO()
        sample_data.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv_buffer.getvalue(),
            file_name="sample_jobs.csv",
            mime="text/csv"
        )
    
    with col2:
        st.info("Download the sample CSV file to see the required format")
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} jobs from {uploaded_file.name}")
            
            # Validate data
            is_valid, errors = validate_csv_data(df)
            
            if not is_valid:
                st.error("❌ CSV validation failed:")
                for error in errors:
                    st.write(f"• {error}")
                st.stop()
            
            # Show data preview
            with st.expander("📋 Data Preview", expanded=True):
                st.dataframe(df.head(10))
                st.caption(f"Showing first 10 of {len(df)} jobs")
            
            # Prediction settings
            st.subheader("⚙️ Prediction Settings")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                batch_size = st.number_input(
                    "Batch Size",
                    min_value=1,
                    max_value=100,
                    value=min(20, len(df)),
                    help="Number of jobs to process in parallel"
                )
            
            with col2:
                add_efficiency = st.checkbox(
                    "Calculate Efficiency",
                    value=True,
                    help="Calculate predicted efficiency for each job"
                )
            
            with col3:
                add_category = st.checkbox(
                    "Add Runtime Category",
                    value=True,
                    help="Categorize jobs by predicted runtime"
                )
            
            # Prediction button
            if st.button("🚀 Run Batch Prediction", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process in batches
                results = []
                total_jobs = len(df)
                processed = 0
                
                start_time = time.time()
                
                try:
                    # Process each job
                    for idx, row in df.iterrows():
                        status_text.text(f"Processing job {processed + 1}/{total_jobs}...")
                        
                        # Prepare job parameters
                        job_params = {
                            'cpus_req': int(row['cpus_req']),
                            'nodes_req': int(row['nodes_req']),
                            'mem_req': float(row['mem_req']),
                            'walltime_req': float(row['walltime_req']),
                            'partition': str(row['partition']),
                            'qos': str(row['qos'])
                        }
                        
                        try:
                            # Make prediction
                            result = client.predict_single(job_params)
                            
                            # Add to results
                            results.append({
                                'job_index': idx + 1,
                                'cpus_req': row['cpus_req'],
                                'nodes_req': row['nodes_req'],
                                'mem_req': row['mem_req'],
                                'walltime_req': row['walltime_req'],
                                'partition': row['partition'],
                                'qos': row['qos'],
                                'predicted_runtime': result.predicted_runtime,
                                'confidence_lower': result.confidence_interval[0],
                                'confidence_upper': result.confidence_interval[1],
                                'runtime_formatted': format_runtime(result.predicted_runtime),
                                'walltime_formatted': format_runtime(row['walltime_req']),
                                'efficiency': calculate_efficiency(
                                    result.predicted_runtime, 
                                    row['walltime_req']
                                ) if add_efficiency else None,
                                'runtime_category': result.runtime_category if add_category else None,
                                'prediction_status': 'success'
                            })
                        except Exception as e:
                            logger.error(f"Prediction failed for job {idx}: {str(e)}")
                            results.append({
                                'job_index': idx + 1,
                                'cpus_req': row['cpus_req'],
                                'nodes_req': row['nodes_req'],
                                'predicted_runtime': 0,
                                'prediction_status': 'failed',
                                'error': str(e)
                            })
                        
                        processed += 1
                        progress_bar.progress(processed / total_jobs)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Calculate statistics
                    elapsed_time = time.time() - start_time
                    success_count = sum(1 for r in results if r['prediction_status'] == 'success')
                    
                    st.success(f"""
                    ✅ Batch prediction completed!
                    - Total jobs: {total_jobs}
                    - Successful: {success_count}
                    - Failed: {total_jobs - success_count}
                    - Processing time: {elapsed_time:.2f} seconds
                    """)
                    
                    # Convert results to DataFrame
                    results_df = pd.DataFrame(results)
                    
                    # Display results
                    st.subheader("📊 Prediction Results")
                    
                    # Summary statistics
                    if success_count > 0:
                        successful_results = results_df[results_df['prediction_status'] == 'success']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            avg_runtime = successful_results['predicted_runtime'].mean()
                            st.metric("Avg Predicted Runtime", format_runtime(avg_runtime))
                        
                        with col2:
                            if add_efficiency:
                                avg_efficiency = successful_results['efficiency'].mean()
                                st.metric("Avg Efficiency", f"{avg_efficiency:.1f}%")
                        
                        with col3:
                            total_runtime = successful_results['predicted_runtime'].sum()
                            st.metric("Total Runtime", format_runtime(total_runtime))
                        
                        with col4:
                            if add_category:
                                categories = successful_results['runtime_category'].value_counts()
                                most_common = categories.idxmax() if not categories.empty else 'N/A'
                                st.metric("Most Common Category", most_common.title())
                    
                    # Results table
                    with st.expander("📋 Detailed Results", expanded=True):
                        # Select columns to display
                        display_columns = [
                            'job_index', 'cpus_req', 'nodes_req', 'mem_req',
                            'runtime_formatted', 'walltime_formatted'
                        ]
                        if add_efficiency:
                            display_columns.append('efficiency')
                        if add_category:
                            display_columns.append('runtime_category')
                        
                        st.dataframe(
                            results_df[display_columns],
                            use_container_width=True
                        )
                    
                    # Visualizations
                    if success_count > 0:
                        st.subheader("📈 Visualizations")
                        
                        # Runtime distribution
                        fig_runtime = px.histogram(
                            successful_results,
                            x='predicted_runtime',
                            nbins=30,
                            title='Distribution of Predicted Runtimes',
                            labels={'predicted_runtime': 'Predicted Runtime (seconds)'}
                        )
                        st.plotly_chart(fig_runtime, use_container_width=True)
                        
                        # Efficiency distribution
                        if add_efficiency:
                            fig_efficiency = px.box(
                                successful_results,
                                x='partition',
                                y='efficiency',
                                title='Efficiency by Partition',
                                labels={'efficiency': 'Efficiency (%)'}
                            )
                            st.plotly_chart(fig_efficiency, use_container_width=True)
                        
                        # Runtime categories
                        if add_category:
                            category_counts = successful_results['runtime_category'].value_counts()
                            fig_categories = px.pie(
                                values=category_counts.values,
                                names=category_counts.index,
                                title='Jobs by Runtime Category'
                            )
                            st.plotly_chart(fig_categories, use_container_width=True)
                    
                    # Download results
                    st.subheader("💾 Download Results")
                    
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Download Full Results (CSV)",
                            data=csv_buffer.getvalue(),
                            file_name=f"predictions_{uploaded_file.name}",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # Summary report
                        summary = f"""
Runtime Prediction Summary Report
================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
File: {uploaded_file.name}

Statistics:
-----------
Total Jobs: {total_jobs}
Successful Predictions: {success_count}
Failed Predictions: {total_jobs - success_count}
Average Predicted Runtime: {format_runtime(successful_results['predicted_runtime'].mean()) if success_count > 0 else 'N/A'}
Total Predicted Runtime: {format_runtime(successful_results['predicted_runtime'].sum()) if success_count > 0 else 'N/A'}
"""
                        if add_efficiency and success_count > 0:
                            summary += f"Average Efficiency: {successful_results['efficiency'].mean():.1f}%\n"
                        
                        st.download_button(
                            label="📄 Download Summary Report",
                            data=summary,
                            file_name=f"summary_{uploaded_file.name.replace('.csv', '.txt')}",
                            mime="text/plain"
                        )
                    
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Batch prediction failed: {str(e)}")
                    logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            logger.error(f"CSV read error: {str(e)}", exc_info=True)
    
    else:
        # No file uploaded - show instructions
        st.info("""
        👆 Upload a CSV file to get started!
        
        **Tips:**
        - Use the sample CSV as a template
        - Ensure all required columns are present
        - Check that numeric values are positive
        - Partition and QoS values should match your HPC system
        """)
