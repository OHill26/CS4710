from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

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

#Create a list of all the features used for clustering.
features = ['studytime', 'failures', 'absences',
            'sex', 'address', 'pstatus', 'schoolsup', 'famsup',
            'internet', 'nursery', 'higher', 'romantic', 'freetime',
            'goout', 'walc', 'dalc'] + \
           [col for col in df.columns if any(prefix in col for prefix in ['mjob_', 'fjob_', 'guardian_', 'reason_'])]


# Set target to predict G3 (final grade)
X = df[features]
y = df['g3']

###############################
### Train a model to predict student performance
# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
#Reference for this: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://repositorium.sdum.uminho.pt/bitstream/1822/8024/1/student.pdf
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

#Scale the data because of different ranges of the data. Without this, the data with high ranges will be prioritized.
X = df[[f for f in features if f not in ['g1', 'g2']]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

### Elbow Method to Determine k #####

wcss = [] # List to store the Within-Cluster Sum of Squares (WCSS) for each k
k_values = range(1, 11) #different k ranges to be used

for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled) #scale the data to the kluster
    wcss.append(kmeans.inertia_) #append the calculated intertia to the list (how tight are the clusters)

plt.figure(figsize=(8, 5))
plt.plot(k_values, wcss, 'bo-') #plot the num clutsers vs wcss (elbow)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

##### K-Means Clustering + PCA Visualization ####
kmeans = KMeans(n_clusters=3, random_state=42) #generate the model
df['cluster'] = kmeans.fit_predict(X_scaled) #fit the model the data

#This code was derived from: https://www.geeksforgeeks.org/principal-component-analysis-with-python/
#Also used: https://setosa.io/ev/principal-component-analysis/
#Drop some non-critical features (only plots 2 and 3 principle data (0-1, 1=2, 2=3)
pca = PCA(n_components=2) 
pca_result = pca.fit_transform(X_scaled)
df['pca1'] = pca_result[:, 0]
df['pca2'] = pca_result[:, 1]

#plot the clusters
#Uses seaborn to create a scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='Set1')
plt.title("K-Means Clusters (Visualized with PCA)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title='Cluster')
plt.show()


#Look at average g3 per cluster
cluster_g3_avg = df.groupby('cluster')['g3'].mean()
print("Average G3 per cluster:\n", cluster_g3_avg)

#Look at different features by cluster
cluster_summary = df.groupby('cluster')[features + ['g3']].mean()
print("Cluster Feature Summary:\n", cluster_summary)

#Box plot of clusters per g3
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='cluster', y='g3', palette='Set2')
plt.title('G3 (Final Grade) Distribution per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Final Grade (G3)')
plt.show()
