# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: vertualforweek2
#     language: python
#     name: python3
# ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from sklearn.decomposition import PCA
from scipy import stats
import sys
import os

# Add the correct relative path to the scripts folder
sys.path.append(os.path.abspath('../scripts'))
sys.path.append(os.path.abspath('../src'))

import cleaned_data_from_db

# Load the dataframe
df = cleaned_data_from_db.df


# Streamlit app title
st.title("Telecom Data Analysis")

# Display the first few rows of the dataframe
st.write("### Data Preview:")
st.dataframe(df.head())

# Interactive slider to select top N handsets
st.sidebar.write("## Handset Analysis")
top_n = st.sidebar.slider("Select the number of top handsets to display", min_value=1, max_value=10, value=5)

# Count handset manufacturers and get the top N
handset_counts = df['Handset Manufacturer'].value_counts()
top_manufacturers = handset_counts.head(top_n)

# Display the top manufacturers in the sidebar
st.sidebar.write(f"### Top {top_n} Handset Manufacturers")
st.sidebar.write(top_manufacturers)

# Select the top 3 manufacturers
top_3_manufacturers = st.sidebar.multiselect(
    "Select manufacturers for detailed analysis",
    options=top_manufacturers.index.tolist(),
    default=top_manufacturers.index[:3].tolist()
)

# Create a dictionary to hold the top 5 handsets for each manufacturer
top_5_handsets = {}

# Iterate over each manufacturer
for manufacturer in top_3_manufacturers:
    # Filter data for the current manufacturer
    manufacturer_data = df[df['Handset Type'].str.contains(manufacturer)]
    
    # Count occurrences of each handset for this manufacturer
    handset_counts = manufacturer_data['Handset Type'].value_counts()
    
    # Get the top 5 handsets
    top_5_handsets[manufacturer] = handset_counts.head(5)

# Plotting the results
for manufacturer, handsets in top_5_handsets.items():
    st.write(f"### Top 5 Handsets for {manufacturer}:")
    st.bar_chart(handsets)

# Bearer ID Analysis
st.sidebar.write("## Bearer ID Analysis")

# Display top 10 user sessions based on Bearer ID
np_xdr_sessions = df['Bearer Id'].value_counts()
top_10_sessions = np_xdr_sessions.head(10)

st.write("### Top 10 Bearer IDs by Session Count:")
st.bar_chart(top_10_sessions)

# Duration of sessions for top users
st.write("### Session Duration for Top 10 Users")
sessions_duration = df[['Dur. (ms)', 'Bearer Id']]
top_10_users = sessions_duration.nlargest(10, 'Dur. (ms)')
st.write(top_10_users)

# Interactive Kernel Density Estimation plot
st.write("### Session Duration KDE Plot")
sns.kdeplot(df['Dur. (ms)'], fill=True)
plt.title('KDE Plot for Session Duration')
st.pyplot()

# Bivariate Analysis
st.write("### Bivariate Analysis: Session Duration vs Total Data Volume")
df['Total_Data_Volume'] = df['Total UL (Bytes)'] + df['Total DL (Bytes)']
sns.scatterplot(x='Dur. (ms)', y='Total_Data_Volume', data=df)
plt.title('Session Duration vs Total Data Volume')
st.pyplot()
