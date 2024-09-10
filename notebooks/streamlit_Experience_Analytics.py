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

# +
import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
import streamlit as st

# Add the correct relative path to the scripts folder
sys.path.append(os.path.abspath('../scripts'))
sys.path.append(os.path.abspath('../src'))

# Import your cleaned data module
import cleaned_data_from_db

# Load the dataframe
df = cleaned_data_from_db.df

# Calculate the new columns
df['TCP_Retransmission'] = df['TCP DL Retrans. Vol (Bytes)'] + df['TCP DL Retrans. Vol (Bytes)']
df['RTT'] = df['Avg RTT DL (ms)'] + df['Avg RTT UL (ms)']
df['Throughput'] = df['Avg Bearer TP DL (kbps)'] + df['Avg Bearer TP UL (kbps)']

# Streamlit App Title
st.title('Telecom User Experience Dashboard')

# Sidebar options for analysis
st.sidebar.header("Options")

# Top N Users based on metrics
top_n = st.sidebar.slider('Select number of top users', min_value=5, max_value=50, value=10)

# Choose to display Top or Bottom users
metric_type = st.sidebar.radio("Choose metric to analyze", ('Top', 'Bottom'))

# Choose which metric to rank by
metric = st.sidebar.selectbox('Select metric to rank users by:', ('TCP_Retransmission', 'RTT', 'Throughput'))

# Top/Bottom Users based on selection
if metric_type == 'Top':
    top_users = df[['MSISDN/Number', metric]].sort_values(by=metric, ascending=False).head(top_n)
else:
    top_users = df[['MSISDN/Number', metric]].sort_values(by=metric, ascending=True).head(top_n)

# Display results
st.write(f"### {metric_type} {top_n} Users by {metric}")
st.dataframe(top_users)

# Cluster Analysis Section
st.write("### Experience Clusters")

# Selecting the experience metrics for clustering
X = df[['TCP_Retransmission', 'RTT', 'Throughput']]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying K-means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
df['Experience_Cluster'] = kmeans.fit_predict(X_scaled)

# Cluster Centroids
centroids = kmeans.cluster_centers_
centroids_df = pd.DataFrame(centroids, columns=['TCP_Retransmission', 'RTT', 'Throughput'])

# Display the cluster centroids
st.write("### Cluster Centroids")
st.dataframe(centroids_df)

# Visualizing the clusters using a pair plot
st.write("### Pair Plot of User Experience Clusters")
sns.pairplot(df, hue='Experience_Cluster', vars=['TCP_Retransmission', 'RTT', 'Throughput'], palette='coolwarm')
st.pyplot()

# Visualize the worst experience cluster (based on highest RTT)
worst_experience_centroid = centroids_df.loc[centroids_df['RTT'].idxmax()]
st.write("### Worst Experience Centroid:")
st.write(worst_experience_centroid)

# Visualize metrics per Handset Type
st.write("### Handset Analysis")
handset_analysis = df.groupby('Handset Type').agg({
    'TCP_Retransmission': 'mean',
    'RTT': 'mean',
    'Throughput': 'mean'
}).reset_index()

# Display the analysis per handset type
st.dataframe(handset_analysis)

