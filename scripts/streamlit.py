import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
# Add the correct relative path to the scripts folder
sys.path.append(os.path.abspath('../scripts'))
sys.path.append(os.path.abspath('../notebooks'))
sys.path.append(os.path.abspath('../src'))

import cleaned_data_from_db


# Load your data
@st.cache_data
def load_data():
    # Replace these with your actual data loading logic
    engagement_df = cleaned_data_from_db
    return engagement_df, satisfaction_df

engagement_df, satisfaction_df = load_data()

st.title("User Experience Dashboard")
st.sidebar.title("Tasks Navigation")

# Task Selection
task = st.sidebar.selectbox("Choose Task", ["Task 1: Experience Analysis", "Task 2: Engagement Analysis", "Task 3: Satisfaction Analysis", "Task 4: Recommendations"])
'''
# Task 1: Experience Analysis
if task == "Task 1: Experience Analysis":
    st.subheader("Experience Analysis")
    
    # Sample Visualization (replace with actual metrics)
    fig = px.scatter(engagement_df, x="RTT", y="Throughput", color="Experience_Cluster", title="Throughput vs RTT by Experience Cluster")
    st.plotly_chart(fig)
    
    # Show statistics
    st.write(engagement_df.describe())

# Task 2: Engagement Analysis
if task == "Task 2: Engagement Analysis":
    st.subheader("Engagement Analysis")
    
    # Engagement Score Visualization
    st.bar_chart(engagement_df['Engagement_Score'])
    
    # Engagement Distribution
    fig = px.histogram(engagement_df, x="Engagement_Score", nbins=50, title="Distribution of Engagement Scores")
    st.plotly_chart(fig)

# Task 3: Satisfaction Analysis
if task == "Task 3: Satisfaction Analysis":
    st.subheader("Satisfaction Analysis")
    
    # Satisfaction score
    st.line_chart(satisfaction_df['Satisfaction_Score'])
    
    # Satisfaction vs Engagement
    fig = px.scatter(satisfaction_df, x="Satisfaction_Score", y="Engagement_Score", title="Satisfaction vs Engagement")
    st.plotly_chart(fig)

# Task 4: Recommendations
if task == "Task 4: Recommendations":
    st.subheader("Final Recommendations")
    
    # Merging the engagement and satisfaction data
    combined_df = pd.merge(engagement_df, satisfaction_df, on="MSISDN/Number")
    
    # Experience Score and Recommendations
    combined_df['Experience_Score'] = (combined_df['Engagement_Score'] + combined_df['Satisfaction_Score']) / 2
    
    # Display Recommendations
    st.write(combined_df[['MSISDN/Number', 'Experience_Score']].sort_values(by='Experience_Score', ascending=False).head())
    
    # Visualize the recommendations
    fig = px.bar(combined_df, x="MSISDN/Number", y="Experience_Score", title="User Recommendations by Experience Score")
    st.plotly_chart(fig)
'''