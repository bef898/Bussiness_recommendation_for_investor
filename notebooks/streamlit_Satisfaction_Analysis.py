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
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Add the correct relative path to the scripts folder
sys.path.append(os.path.abspath('../scripts'))
sys.path.append(os.path.abspath('../src'))
# -

df = pd.read_csv('aggregated_data.csv')

df

df_top =df.head(10)
df_top

df_least =df.tail(10)
df_least

data = pd.read_csv('aggregated_data_for_experiance.csv')

df.head()

# +
# Selecting the experience metrics for clustering
X = df[['Bearer Id', 'Dur. (ms)', 'Total_Traffic']]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# +
# Apply K-means clustering (assuming 3 clusters for low, medium, high engagement)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data)



# +
# Applying K-means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
data['engagment_cluster'] = kmeans.fit_predict(X_scaled)

# Display the clustering results
centroids_df =df[['Bearer Id', 'Dur. (ms)', 'Total_Traffic']]
centroids = kmeans.cluster_centers_
print("engagment_cluster:\n", centroids)


# Get the centroids of the clusters
#engagement_centroids = kmeans.cluster_centers_

# Convert centroids to DataFrame for analysis
#centroids_df = pd.DataFrame(engagement_centroids, columns=['Bearer Id', 'Dur. (ms)', 'Total_Traffic'])




# -

# Identify the worst engagement cluster (users with lowest sessions, lowest duration)
worst_engagement_centroid = centroids_df.loc[centroids_df['Total_Traffic'].idxmin()]

worst_engagement_centroid


# +
# Function to calculate Euclidean distance (Engagement Score) for each user
def calculate_engagement_score(row, centroid):
    # Row values are number of sessions, pages per session, and session duration
    user_metrics = np.array([row['Bearer Id'], row['Dur. (ms)'], row['Total_Traffic']])
    
    # Calculate Euclidean distance
    distance = np.sqrt(np.sum((user_metrics - worst_engagement_centroid) ** 2))
    return distance

# Create DataFrame for engagement data (replace with actual columns)
df_engagement = pd.DataFrame(df, columns=['Bearer Id', 'Dur. (ms)', 'Total_Traffic'])

# Apply the function to each row to calculate the engagement score
df_engagement['Engagement_Score'] = df_engagement.apply(calculate_engagement_score, axis=1, centroid=centroids_df)

# Display the DataFrame with Engagement Score
print(df_engagement)
# -

df_engagement

# +
# Selecting the experience metrics for clustering
X = data[['TCP_Retransmission', 'RTT', 'Throughput']]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# +
# Applying K-means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
data['Experience_Cluster'] = kmeans.fit_predict(X_scaled)

# Display the clustering results
print(data[['MSISDN/Number', 'TCP_Retransmission', 'RTT', 'Throughput', 'Experience_Cluster']])
centroids = kmeans.cluster_centers_
print("Cluster Centroids:\n", centroids)


# +
# Convert centroids to a DataFrame for easy analysis
centroids_df = pd.DataFrame(centroids, columns=['TCP_Retransmission', 'RTT', 'Throughput'])

# Identify the worst experience cluster based on the metrics (highest RTT and TCP retransmission)
worst_experience_centroid = centroids_df.loc[centroids_df['RTT'].idxmax()]
print("Worst Experience Centroid:\n", worst_experience_centroid)

# +
worst_experience_centroid = np.array(worst_experience_centroid)

# Calculate Euclidean distance (Experience Score) for each user
def calculate_experience_score(row, centroid):
    # Row values are TCP Retransmission, RTT, Throughput
    user_metrics = np.array([row['TCP_Retransmission'], row['RTT'], row['Throughput']])
    
    # Calculate Euclidean distance
    distance = np.sqrt(np.sum((user_metrics - centroid) ** 2))
    return distance

# Apply the function to each row to calculate the experience score
data['Experience_Score'] = data.apply(calculate_experience_score, axis=1, centroid=worst_experience_centroid)

# Display the updated DataFrame with the new Experience_Score column
experiance_score =data[['MSISDN/Number', 'TCP_Retransmission', 'RTT', 'Throughput', 'Experience_Score']]


# -

experiance_score.head(10)

# +
# Combine the engagement and experience scores into one DataFrame
satisfaction_df = pd.concat([df_engagement, experiance_score], axis=1)

# Calculate the Satisfaction Score as the average of Engagement Score and Experience Score
satisfaction_df['Satisfaction_Score'] = (satisfaction_df['Engagement_Score'] + satisfaction_df['Experience_Score']) / 2

print("Satisfaction Scores:\n", satisfaction_df[['Engagement_Score', 'Experience_Score', 'Satisfaction_Score']])

# -

satisfaction_df['Satisfaction_Score']

# +
# Engagement and experience metrics combined as features (X) and satisfaction score as target (y)
X = np.hstack([df_engagement  , experiance_score  ])  # Combine engagement and experience metrics
y = satisfaction_df  # Satisfaction scores as target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict satisfaction scores on the test set
y_pred = model.predict(X_test)

# Display predicted satisfaction scores and actual satisfaction scores
print("Predicted Satisfaction Scores:", y_pred)
print("Actual Satisfaction Scores:", y_test)


# +
from sklearn.metrics import mean_squared_error, r2_score

# Calculate MSE and R²
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

