# first we import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load the dataset and display
df = pd.read_csv('mall_customers.csv')
print(df.head())

# Select features relevant to purchase history for clustering
# In this case, we use 'Age', 'Annual Income', and 'Spending Score (1-100)'
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Normalize the data (important for K-Means) 
#Features with larger scales (e.g., income in thousands vs. spending 
#score in a range of 1-100) will dominate the distance calculation. 
#As a result, the clustering may be biased toward features with larger 
#values, even though other features might be equally important.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#determine the optimal number of clusters using the elbow method
inertia = []
K = range(1, 20)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
# Plotting the Elbow Curve
plt.figure(figsize=(12, 9))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()


# Choose the number of clusters (K) - by the elbow method as an example
k = 5  
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
kmeans.fit(X_scaled)
labels=kmeans.fit_predict(X)

#add the cluster label to the dataframe
df['Cluster'] =labels
#display the first few rows of the dataframe with the cluster labels
df.head(100)

# Add cluster labels back to the original dataset
df['Cluster'] = kmeans.labels_

# Visualization of the clusters
plt.figure(figsize=(10, 8))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
plt.xlabel('Scaled Annual Income')
plt.ylabel('Scaled Spending Score')
plt.title('Clusters of Customers')
plt.show()


kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)


#Visualize the clusters in a 3D scatter plot (Age, Annual Income, Spending Score)
fig = plt.figure(figsize=(20,14))
ax = fig.add_subplot(111, projection='3d')

#Visualization of the cluster in 2d scatter plot
ax.scatter(df['Age'], df['Annual Income (k$)'], df['Spending Score (1-100)'], 
           c=df['Cluster'], cmap='viridis', s=100, edgecolor='k')

ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
plt.title('K-Means Clustering of Retail Store Customers')
plt.show()
sns.set(style='whitegrid')
fig, axs=plt.subplots(1,3,figsize=(20,5))
#Visualize the cluster in histogram
sns.histplot(data=df,x='Age',kde=True,color='blue',ax=axs[0])
sns.histplot(data=df, x='Annual Income (k$)', kde=True, color='green', ax=axs[1])
sns.histplot(data=df, x='Spending Score (1-100)', kde=True, color='red', ax=axs[2])








