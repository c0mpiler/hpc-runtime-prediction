"""
Single Job Prediction Page
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from utils.grpc_client import format_runtime

def show_single_prediction(client):
    """Display single job prediction interface."""
    
    st.header("🎯 Single Job Prediction")
    st.markdown("Enter job parameters to predict runtime.")
    
    # Job parameters form
    with st.form("single_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            nodes = st.number_input("Nodes Required", min_value=1, max_value=100, value=1)
            cpus = st.number_input("CPUs Required", min_value=1, max_value=256, value=16)
            mem_gb = st.number_input("Memory (GB)", min_value=1, max_value=1000, value=32)
        
        with col2:
            walltime_hours = st.number_input("Requested Walltime (hours)", min_value=0.1, max_value=168.0, value=1.0, step=0.5)
            partition = st.selectbox("Partition", ["compute", "gpu", "debug", "standard", "bigmem"])
            qos = st.selectbox("QoS", ["normal", "high", "low", "debug"])
        
        with col3:
            gpus = st.number_input("GPUs Required", min_value=0, max_value=8, value=0)
            account = st.text_input("Account", value="default")
            username = st.text_input("Username", value="user")
        
        submitted = st.form_submit_button("🚀 Predict Runtime", type="primary")
    
    if submitted:
        try:
            # Prepare job parameters
            job_params = {
                'nodes_req': nodes,
                'cpus_req': cpus,
                'mem_req': mem_gb * 1024,  # Convert GB to MB
                'walltime_req': int(walltime_hours * 3600),  # Convert hours to seconds
                'partition': partition,
                'qos': qos,
                'gpus_req': gpus,
                'account': account,
                'username': username,
                'submit_time': int(datetime.now().timestamp())
            }
            
            # Make prediction
            with st.spinner("🔮 Making prediction..."):
                prediction = client.predict_single(job_params)
            
            if prediction:
                # Display results
                st.success("✅ Prediction completed successfully!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Predicted Runtime",
                        format_runtime(int(prediction.predicted_runtime)),
                        help="Predicted actual execution time"
                    )
                    
                    # Display confidence score
                    st.metric(
                        "Confidence",
                        f"{prediction.confidence_score:.1%}",
                        help="Model confidence in prediction"
                    )
                
                with col2:
                    # Display confidence interval
                    confidence_range = prediction.confidence_interval[1] - prediction.confidence_interval[0]
                    st.metric(
                        "Confidence Interval",
                        f"±{confidence_range/2:.0f}s",
                        f"[{format_runtime(int(prediction.confidence_interval[0]))} - {format_runtime(int(prediction.confidence_interval[1]))}]",
                        help="95% confidence interval for prediction"
                    )
                    
                    # Display model version
                    st.metric(
                        "Model Version",
                        prediction.model_version,
                        help="Version of the prediction model"
                    )
                
                # Show prediction details
                with st.expander("📊 Prediction Details"):
                    # Convert to dict for display
                    pred_dict = prediction.to_dict()
                    details_df = pd.DataFrame([
                        {"Metric": "Runtime Category", "Value": prediction.runtime_category},
                        {"Metric": "Prediction Time", "Value": prediction.prediction_time.strftime("%Y-%m-%d %H:%M:%S")},
                        {"Metric": "Features Used", "Value": len(prediction.features_used)}
                    ])
                    st.dataframe(details_df, use_container_width=True)
                
                # Efficiency recommendation based on predicted vs requested time
                efficiency = (prediction.predicted_runtime / job_params['walltime_req']) * 100
                if efficiency < 50:
                    st.info("💡 Your requested walltime is significantly higher than predicted. Consider reducing it to improve queue priority.")
                elif efficiency > 90:
                    st.warning("⚠️ Predicted runtime is close to requested walltime. Consider adding a buffer to avoid job timeout.")
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.info("Please check your input parameters and try again.")
