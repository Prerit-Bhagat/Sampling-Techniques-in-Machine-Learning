import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv('Creditcard_data.csv')

# Split data into features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Random Undersampling
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_undersampled, y_undersampled = undersampler.fit_resample(X, y)
print("Undersampled Data Class Distribution:")
print(pd.Series(y_undersampled).value_counts())

# Random Oversampling
oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_oversampled, y_oversampled = oversampler.fit_resample(X, y)
print("\nOversampled Data Class Distribution:")
print(pd.Series(y_oversampled).value_counts())

# SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)
print("\nSMOTE Data Class Distribution:")
print(pd.Series(y_smote).value_counts())

# Cluster Sampling
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)
X_clustered = X.copy()
X_clustered['Cluster'] = clusters
X_clustered_sampled = X_clustered.groupby('Cluster').sample(n=50, random_state=42)
y_clustered_sampled = y[X_clustered_sampled.index]
print("\nCluster-Sampled Data Class Distribution:")
print(pd.Series(y_clustered_sampled).value_counts())

# Initialize Random Forest Classifier
model = RandomForestClassifier(random_state=42)

# Function to train, predict and evaluate model
def train_and_evaluate(X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred)

# Train and evaluate on the original data (without sampling)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nOriginal Data Classification Report:")
print(train_and_evaluate(X_train, X_test, y_train, y_test))

# Train and evaluate on Undersampled Data
X_train_under, X_test_under, y_train_under, y_test_under = train_test_split(X_undersampled, y_undersampled, test_size=0.2, random_state=42)
print("\nUndersampled Data Classification Report:")
print(train_and_evaluate(X_train_under, X_test_under, y_train_under, y_test_under))

# Train and evaluate on Oversampled Data
X_train_over, X_test_over, y_train_over, y_test_over = train_test_split(X_oversampled, y_oversampled, test_size=0.2, random_state=42)
print("\nOversampled Data Classification Report:")
print(train_and_evaluate(X_train_over, X_test_over, y_train_over, y_test_over))

# Train and evaluate on SMOTE Data
X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)
print("\nSMOTE Data Classification Report:")
print(train_and_evaluate(X_train_smote, X_test_smote, y_train_smote, y_test_smote))

# Train and evaluate on Cluster-Sampled Data
X_train_cluster, X_test_cluster, y_train_cluster, y_test_cluster = train_test_split(X_clustered_sampled.drop('Cluster', axis=1), y_clustered_sampled, test_size=0.2, random_state=42)
print("\nCluster-Sampled Data Classification Report:")
print(train_and_evaluate(X_train_cluster, X_test_cluster, y_train_cluster, y_test_cluster))
