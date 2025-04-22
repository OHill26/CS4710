from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def initialize_centroids(X, k):
    indices = np.random.choice(X.shape[0], k, replace=False) #Randomly pick 3 data points from the scaled features
    return X[indices]

def euclidean_distance(point1, point2):
    squared_diff = [(p1 - p2) ** 2 for p1, p2 in zip(point1, point2)] #find euclidean distance using formula
    return sum(squared_diff) ** 0.5 #take square root of total

#Compute distance from centroids
def compute_distances(X, centroids):
    distances = []
    for point in X:     ##Loop through students and their features
        point_distances = []
        for centroid in centroids:  ##Loop through centroids 
            d = euclidean_distance(point, centroid)  #find distance from student to centroid
            point_distances.append(d)      #add distance to array
        distances.append(point_distances) #add all distances to array
    return np.array(distances)

#Assign student to a cluster
def assign_clusters(distances):
    return np.argmin(distances, axis=1)

##Update the centroids based on clusters
def update_centroids(X, labels, k):
    new_centroids = []
    for i in range(k):
        # Check if there are points in this cluster
        if np.sum(labels == i) == 0:
            print(f"Warning: Cluster {i} is empty. Reinitializing centroid.")
            new_centroids.append(X[np.random.choice(X.shape[0])])  # Randomly pick a point as the new centroid
        else:
            new_centroids.append(X[labels == i].mean(axis=0))  # Calculate mean of the points in the cluster
    return np.array(new_centroids)

def kmeans(X, k, max_iters=100, tol=1e-4):
    X = np.array(X) #convert Students to numpy array to make calculations easier
    centroids = initialize_centroids(X, k) #Create initial random centroids

    for i in range(max_iters): 
        old_centroids = centroids.copy() #Save old centroids that will be used to check if the new centroids are being changed, if same -> stop.

        #Assign points to nearest centroid
        distances = compute_distances(X, centroids)
        labels = assign_clusters(distances)

        #Update centroids based on the new points assignment
        centroids = update_centroids(X, labels, k)

        #Check for convergence
        if np.all(np.linalg.norm(centroids - old_centroids, axis=1) < tol):
            print(f"Converged in {i} iterations.")
            break #break if converged

    return labels, centroids

##Find 
def calculate_wcss(X, labels, centroids):
    wcss = 0 
    for i, point in enumerate(X): ##loop through student data features
        center = centroids[labels[i]] ##get cluster
        wcss += np.sum((point - center) ** 2) ##find distance 
    return wcss

def run_elbow_plot(X_scaled, k_range=range(1, 11)):
    wcss = []
    for k in k_range:
        labels, centroids = kmeans(X_scaled, k)
        score = calculate_wcss(X_scaled, labels, centroids)
        wcss.append(score)

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, wcss, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()

def cluster_subset(df, feature_list, subset_name, k=3):
    print(f"\n--- {subset_name} Subset ---")
    X_sub = df[feature_list].copy() #Grab relevant feautres
    scaler = StandardScaler() #scale data so no features dominate
    X_scaled = scaler.fit_transform(X_sub)

    labels, centroids = kmeans(X_scaled, k) #runs k-means clustering
    sil_score = silhouette_score(X_scaled, labels) #Find reliability of kmean model
    print(f"Silhouette Score: {sil_score:.3f}")

    cluster_col = f'{subset_name.lower()}_cluster' #set subset name neatly for printing
    df[cluster_col] = labels #add assignment clusters to array for visualization

    #print out grade for interpretation
    if 'g3' in df.columns:
        print("G3 by cluster:")
        print(df.groupby(cluster_col)['g3'].mean())
    
    summarize_clusters(df, cluster_col, feature_list) #print statistics

    # PCA plot
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df[f'{subset_name.lower()}_pca1'] = pca_result[:, 0]
    df[f'{subset_name.lower()}_pca2'] = pca_result[:, 1]

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=f'{subset_name.lower()}_pca1', y=f'{subset_name.lower()}_pca2',
                    hue=cluster_col, palette='Set1')
    plt.title(f"{subset_name} Clustering (PCA)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title='Cluster')
    plt.show()

def summarize_clusters(df, cluster_col, features, target='g3'):
    summary = df.groupby(cluster_col)[features + [target]].mean()
    print(f"\nSummary for '{cluster_col}':")
    print(summary)
    return summary

# fetch dataset 
student_performance = fetch_ucirepo(id=320) 

# Copy features and targets separately
features_df = student_performance.data.features.copy()
targets_df = student_performance.data.targets.copy()

# Merge them into one DataFrame
df = pd.concat([features_df, targets_df], axis=1)

# Normalize column names (make lowercase, replace spaces with underscores)
df.columns = df.columns.str.lower().str.replace(" ", "_")  


#####  Label Encode Binary Columns #####

#gather binary data to change to 0 or 1 so I can cluster it
binary_cols = [
    'sex', 'school', 'address', 'pstatus', 'schoolsup', 'famsup',
    'activities', 'paid', 'internet', 'nursery', 'higher', 'romantic'
]

#Convert the categorical data to numerical data (0 or 1)
label_enc = LabelEncoder()
for col in binary_cols:
    if col in df.columns:
        df.loc[:, col] = label_enc.fit_transform(df[col])  

#Turn nominal data into numerical data, when one is true, that category is set to 1, the rest are set to 0
nominal_cols = ['mjob', 'fjob', 'guardian', 'reason']
df = pd.get_dummies(df, columns=[col for col in nominal_cols if col in df.columns], drop_first=True)

academic_features = ['studytime', 'failures', 'absences']
family_features = ['address', 'pstatus', 'famsup'] + [col for col in df.columns if col.startswith(('mjob_', 'fjob_', 'guardian_', 'reason_'))]
behavioral_features = ['freetime', 'goout', 'walc', 'dalc', 'romantic']
support_features = ['internet', 'paid', 'nursery', 'higher', 'activities', 'schoolsup']

#A dictionary that holds the different subsets of data
#Going to be used to create clusters/kmeans model on related student features
#Did this to improve silhouette scores
subsets = {
    'Academic': academic_features,
    'Family': family_features,
    'Behavioral': behavioral_features,
    'Support': support_features
}

for name, features in subsets.items():
    cluster_subset(df, features, name)


##Clustering all the features
full_features = academic_features + family_features + behavioral_features + support_features #combine all features
cluster_subset(df, full_features, "AllFeatures", k=3) #Call clustering functions on all features list




