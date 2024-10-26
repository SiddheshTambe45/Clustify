# dataupload/management/commands/analyze_data.py
import pandas as pd
from sklearn.cluster import KMeans
from django.core.management.base import BaseCommand
from dataupload.models import JSONData
import matplotlib.pyplot as plt
import os

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from django.core.management.base import BaseCommand
from dataupload.models import JSONData,NumericData
import matplotlib.pyplot as plt
import os

# class Command(BaseCommand):
#     help = 'Performs clustering on JSONData model and saves a plot'
#
#     def handle(self, *args, **kwargs):
#         # Load data from the model
#         data = JSONData.objects.all().values('name', 'age')
#         df = pd.DataFrame(data)
#
#         # Remove any nulls (optional)
#         df = df.dropna()
#
#         # Extract features for clustering
#         features = df[['age']]
#
#         # Perform K-means clustering
#         kmeans = KMeans(n_clusters=3)  # Set desired number of clusters
#         df['cluster'] = kmeans.fit_predict(features)
#
#         # Print cluster centers
#         print("Cluster centers:")
#         print(kmeans.cluster_centers_)
#
#         # Plotting the results
#         plt.figure(figsize=(10, 6))
#         plt.scatter(df['age'], [0] * len(df), c=df['cluster'], cmap='viridis', marker='o', edgecolor='k', s=100)
#         plt.scatter(kmeans.cluster_centers_, [0] * len(kmeans.cluster_centers_), c='red', s=200, marker='X',
#                     label='Centroids')
#         plt.title('K-means Clustering of Age Data')
#         plt.xlabel('Age')
#         plt.yticks([])  # Hide y-axis ticks
#         plt.legend()
#
#         # Save the plot
#         plot_dir = 'dataupload/plots'
#         os.makedirs(plot_dir, exist_ok=True)  # Create directory if it doesn't exist
#         plt.savefig(os.path.join(plot_dir, 'clustering_plot.png'))
#         plt.close()  # Close the plot to free memory

        # Optional: save clusters back to the database
        # for index, row in df.iterrows():
        #     json_data_entry = JSONData.objects.get(id=row['id'])
        #     json_data_entry.json_content['cluster'] = int(row['cluster'])
        #     json_data_entry.save(update_fields=['json_content'])


class Command(BaseCommand):
    help = 'Performs clustering on JSONData model and saves a plot'

    def handle(self, *args, **kwargs):
        # Load data from the model
        data = NumericData.objects.all().values('Age', 'Gender')  # Ensure 'purchase_history' is included
        df = pd.DataFrame(data)

        # Remove any nulls (optional)
        df = df.dropna()

        # Prepare features for clustering
        # Convert Purchase History to numerical representation
        # Assuming Purchase History is a dictionary, you may want to extract relevant metrics
        # df['purchase_history_value'] = df['Purchase_History'].apply(
        #     lambda x: sum(x.values()) if isinstance(x, dict) else 0)



        # Extract features for clustering
        features = df[['Age', 'gender_value']]

        # Standardizing the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=3)  # Set desired number of clusters
        df['cluster'] = kmeans.fit_predict(scaled_features)

        # Print cluster centers
        print("Cluster centers:")
        print(kmeans.cluster_centers_)

        # Plotting the results
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(df['Age'], df['gender_value'], c=df['cluster'], cmap='viridis', marker='o',
                              edgecolor='k', s=100)

        # Plot cluster centroids
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')

        plt.title('K-means Clustering of Age and Purchase History')
        plt.xlabel('Age')
        plt.ylabel('Total Purchase History Value')  # Adjust label based on your interpretation
        plt.legend()

#         # Save the plot
#         plot_dir = 'dataupload/plots'
#         os.makedirs(plot_dir, exist_ok=True)  # Create directory if it doesn't exist
#         plt.savefig(os.path.join(plot_dir, 'clustering_plot.png'))
#         plt.close()  # Close the plot to free memory


import ast  # For safe list conversion if stored as strings
# from dataupload.models import JSONData
# from sklearn.preprocessing import MultiLabelBinarizer
# from scipy.cluster.hierarchy import linkage, dendrogram
# import matplotlib.pyplot as plt
#
#
# class Command(BaseCommand):
#     help = 'Performs hierarchical clustering on JSONData model and saves a dendrogram plot'
#
#     def handle(self, *args, **kwargs):
#         # Load data from the model
#         data = JSONData.objects.all().values('Age', 'Purchase_History')
#         df = pd.DataFrame(data)
#
#         # Drop any nulls to clean the data
#         df = df.dropna()
#
#         # Use MultiLabelBinarizer to handle lists in Purchase_History
#         mlb = MultiLabelBinarizer()
#         purchase_history_encoded = mlb.fit_transform(df['Purchase_History'])
#
#         # Combine 'Age' with the encoded purchase history
#         age = df[['Age']].values
#         features = pd.concat([pd.DataFrame(age, columns=['Age']), pd.DataFrame(purchase_history_encoded)], axis=1)
#
#         # Perform hierarchical clustering
#         Z = linkage(features, method='ward')  # 'ward' minimizes variance within clusters
#
#         # Plot dendrogram
#         plt.figure(figsize=(10, 7))
#         dendrogram(Z, labels=df['Age'].values, orientation='top', leaf_rotation=90)
#         plt.title('Hierarchical Clustering Dendrogram')
#         plt.xlabel('Sample index')
#         plt.ylabel('Distance')
#
#         # Save the plot
#         plot_dir = 'dataupload/plots'
#         os.makedirs(plot_dir, exist_ok=True)
#         plt.savefig(os.path.join(plot_dir, 'hierarchical_clustering_dendrogram.png'))
#         plt.close()  # Close the plot to free memory
#
#         self.stdout.write(self.style.SUCCESS('Hierarchical clustering completed and dendrogram saved.'))