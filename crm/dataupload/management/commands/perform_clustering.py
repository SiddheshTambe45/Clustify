# # from django.core.management.base import BaseCommand
# # from dataupload.models import NumericData  # Replace with your app name
# # import pandas as pd
# # import numpy as np
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.cluster import KMeans
# # import matplotlib.pyplot as plt
# # import os
# # from datetime import datetime
# #
# #
# # class Command(BaseCommand):
# #     help = 'Perform K-means clustering on Age and Income Level'
# #
# #     def handle(self, *args, **kwargs):
# #         # Load data
# #         data = NumericData.objects.all().values('Age', 'Income_Level')
# #         df = pd.DataFrame(data)
# #
# #         # Clean data
# #         df = df.dropna()
# #         df = df[(df['Age'] > 0) & (df['Income_Level'] > 0)]
# #
# #         if len(df) == 0:
# #             self.stdout.write(self.style.ERROR('No valid data found'))
# #             return
# #
# #         # Scale features
# #         scaler = StandardScaler()
# #         scaled_features = scaler.fit_transform(df[['Age', 'Income_Level']])
# #
# #         # Perform clustering
# #         kmeans = KMeans(n_clusters=3, random_state=42)
# #         df['Cluster'] = kmeans.fit_predict(scaled_features)
# #
# #         # Get cluster centers
# #         centers = scaler.inverse_transform(kmeans.cluster_centers_)
# #
# #         # Create plot
# #         plt.figure(figsize=(12, 8))
# #
# #         # Scatter plot with different colors for each cluster
# #         scatter = plt.scatter(df['Age'],
# #                               df['Income_Level'],
# #                               c=df['Cluster'],
# #                               cmap='viridis',
# #                               alpha=0.6,
# #                               s=100)
# #
# #         # Plot cluster centers
# #         plt.scatter(centers[:, 0],
# #                     centers[:, 1],
# #                     c='red',
# #                     marker='X',
# #                     s=200,
# #                     label='Centroids')
# #
# #         # Add labels and title
# #         plt.title('Customer Segments: Age vs Income Level', fontsize=14)
# #         plt.xlabel('Age', fontsize=12)
# #         plt.ylabel('Income Level', fontsize=12)
# #
# #         # Add colorbar and legend
# #         plt.colorbar(scatter, label='Cluster')
# #         plt.legend()
# #
# #         # Add grid
# #         plt.grid(True, linestyle='--', alpha=0.7)
# #
# #         # Save plot
# #         plot_dir = 'dataupload/plots'
# #         os.makedirs(plot_dir, exist_ok=True)
# #
# #         # Save with timestamp
# #         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# #         plot_path = os.path.join(plot_dir, f'clustering_plot_{timestamp}.png')
# #         plt.savefig(plot_path, dpi=300, bbox_inches='tight')
# #         plt.close()
# #
# #         # Print summary
# #         self.stdout.write(self.style.SUCCESS(f'Clustering complete! Plot saved to: {plot_path}'))
# #
# #         # Print cluster statistics
# #         for cluster in range(3):
# #             cluster_data = df[df['Cluster'] == cluster]
# #             self.stdout.write(f'\nCluster {cluster} statistics:')
# #             self.stdout.write(f'Count: {len(cluster_data)}')
# #             self.stdout.write(f'Average Age: {cluster_data["Age"].mean():.2f}')
# #             self.stdout.write(f'Average Income: {cluster_data["Income_Level"].mean():.2f}')
#
#
# from django.core.management.base import BaseCommand
# from dataupload.models import NumericData
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# import os
# from datetime import datetime
# import itertools
#
#
# class Command(BaseCommand):
#     help = 'Perform K-means clustering on all numeric attributes'
#
#     def create_clustering_plot(self, df, feature1, feature2, scaler, timestamp, plot_dir):
#         """Create and save clustering plot for two features"""
#         # Clean data for these features
#         plot_df = df[[feature1, feature2]].dropna()
#         plot_df = plot_df[(plot_df[feature1] != 0) & (plot_df[feature2] != 0)]
#
#         if len(plot_df) < 3:
#             self.stdout.write(self.style.WARNING(
#                 f'Insufficient data for clustering {feature1} vs {feature2}'))
#             return None
#
#         # Scale features
#         scaled_features = scaler.fit_transform(plot_df)
#
#         # Perform clustering
#         kmeans = KMeans(n_clusters=3, random_state=42)
#         plot_df['Cluster'] = kmeans.fit_predict(scaled_features)
#
#         # Get cluster centers
#         centers = scaler.inverse_transform(kmeans.cluster_centers_)
#
#         # Create plot
#         plt.figure(figsize=(12, 8))
#
#         # Scatter plot
#         scatter = plt.scatter(plot_df[feature1],
#                               plot_df[feature2],
#                               c=plot_df['Cluster'],
#                               cmap='viridis',
#                               alpha=0.6,
#                               s=100)
#
#         # Plot centers
#         plt.scatter(centers[:, 0],
#                     centers[:, 1],
#                     c='red',
#                     marker='X',
#                     s=200,
#                     label='Centroids')
#
#         # Labels and title
#         plt.title(f'Customer Segments: {feature1} vs {feature2}', fontsize=14)
#         plt.xlabel(feature1, fontsize=12)
#         plt.ylabel(feature2, fontsize=12)
#
#         # Colorbar and legend
#         plt.colorbar(scatter, label='Cluster')
#         plt.legend()
#
#         # Grid
#         plt.grid(True, linestyle='--', alpha=0.7)
#
#         # Save plot
#         plot_name = f'clustering_{feature1}_{feature2}_{timestamp}.png'
#         plot_path = os.path.join(plot_dir, plot_name)
#         plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#         plt.close()
#
#         # Calculate cluster statistics
#         stats = []
#         for cluster in range(3):
#             cluster_data = plot_df[plot_df['Cluster'] == cluster]
#             stats.append({
#                 'cluster': cluster,
#                 'count': len(cluster_data),
#                 f'avg_{feature1}': cluster_data[feature1].mean(),
#                 f'avg_{feature2}': cluster_data[feature2].mean()
#             })
#
#         return {'plot_path': plot_path, 'statistics': stats}
#
#     def handle(self, *args, **kwargs):
#         # Define numeric fields to analyze
#         numeric_fields = [
#             'Age',
#             'Gender',
#             'Region',
#             'City',
#             'Feedback',
#             'Income_Level',
#         ]
#
#         # Load data
#         data = NumericData.objects.all().values(*numeric_fields)
#         df = pd.DataFrame(data)
#
#         # Create plots directory
#         plot_dir = 'dataupload/plots'
#         os.makedirs(plot_dir, exist_ok=True)
#
#         # Generate timestamp
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#
#         # Create scaler
#         scaler = StandardScaler()
#
#         # Generate all possible pairs of features
#         feature_pairs = list(itertools.combinations(numeric_fields, 2))
#
#         # Process each pair
#         for feature1, feature2 in feature_pairs:
#             self.stdout.write(f'\nProcessing clustering for {feature1} vs {feature2}...')
#
#             result = self.create_clustering_plot(
#                 df,
#                 feature1,
#                 feature2,
#                 scaler,
#                 timestamp,
#                 plot_dir
#             )
#
#             if result:
#                 self.stdout.write(self.style.SUCCESS(
#                     f'Plot saved to: {result["plot_path"]}'))
#
#                 # Print cluster statistics
#                 for stat in result['statistics']:
#                     self.stdout.write(
#                         f'\nCluster {stat["cluster"]} statistics:'
#                         f'\nCount: {stat["count"]}'
#                         f'\nAverage {feature1}: {stat[f"avg_{feature1}"]:.2f}'
#                         f'\nAverage {feature2}: {stat[f"avg_{feature2}"]:.2f}'
#                     )
#
#         self.stdout.write(self.style.SUCCESS('\nClustering analysis complete!'))


# management/commands/perform_clustering.py
from django.core.management.base import BaseCommand
from dataupload.models import NumericData, ClusterAnalysis, ClusterDetails, Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
from datetime import datetime
import itertools


class Command(BaseCommand):
    help = 'Perform and store K-means clustering analysis on all numeric attributes'

    def create_clustering_analysis(self, df, feature1, feature2, scaler, timestamp, plot_dir):
        """Perform clustering analysis and store results"""
        # Clean data
        plot_df = df[['CustomerID', feature1, feature2]].dropna()
        plot_df = plot_df[(plot_df[feature1] != 0) & (plot_df[feature2] != 0)]

        if len(plot_df) < 3:
            self.stdout.write(self.style.WARNING(
                f'Insufficient data for clustering {feature1} vs {feature2}'))
            return None

        # Scale features for clustering
        features_for_clustering = plot_df[[feature1, feature2]]
        scaled_features = scaler.fit_transform(features_for_clustering)

        # Perform clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        plot_df['Cluster'] = kmeans.fit_predict(scaled_features)

        # Calculate silhouette score
        sil_score = silhouette_score(scaled_features, plot_df['Cluster'])

        # Create and save plot
        plt.figure(figsize=(12, 8))

        # Get cluster centers
        centers = scaler.inverse_transform(kmeans.cluster_centers_)

        # Create scatter plot
        scatter = plt.scatter(plot_df[feature1],
                              plot_df[feature2],
                              c=plot_df['Cluster'],
                              cmap='viridis',
                              alpha=0.6,
                              s=100)

        # Plot centers
        plt.scatter(centers[:, 0],
                    centers[:, 1],
                    c='red',
                    marker='X',
                    s=200,
                    label='Centroids')

        plt.title(f'Customer Segments: {feature1} vs {feature2}\nSilhouette Score: {sil_score:.3f}',
                  fontsize=14)
        plt.xlabel(feature1, fontsize=12)
        plt.ylabel(feature2, fontsize=12)
        plt.colorbar(scatter, label='Cluster')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Save plot
        plot_name = f'clustering_{feature1}_{feature2}_{timestamp}.png'
        plot_path = os.path.join(plot_dir, plot_name)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # plot_name = f'clustering_{feature1}_{feature2}_{timestamp}.png'
        # plot_path = os.path.join(plot_dir, plot_name)
        # plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        # plt.close()

        # Create Image record
        Image.objects.create(
            title=f'Clustering: {feature1} vs {feature2}',
            image=f'plots/{plot_name}'
        )

        # Create ClusterAnalysis record
        analysis = ClusterAnalysis.objects.create(
            feature1=feature1,
            feature2=feature2,
            plot_path=plot_path,
            total_samples=len(plot_df),
            silhouette_score=sil_score
        )

        # Store details for each cluster
        for cluster_num in range(3):
            cluster_data = plot_df[plot_df['Cluster'] == cluster_num]
            cluster_scaled_data = scaled_features[plot_df['Cluster'] == cluster_num]

            # Calculate cluster density (average distance to centroid)
            centroid = kmeans.cluster_centers_[cluster_num]
            distances = np.linalg.norm(cluster_scaled_data - centroid, axis=1)
            avg_distance = np.mean(distances) if len(distances) > 0 else 0

            ClusterDetails.objects.create(
                analysis=analysis,
                cluster_number=cluster_num,
                sample_count=len(cluster_data),
                feature1_mean=cluster_data[feature1].mean(),
                feature1_std=cluster_data[feature1].std(),
                feature2_mean=cluster_data[feature2].mean(),
                feature2_std=cluster_data[feature2].std(),
                centroid_feature1=centers[cluster_num][0],
                centroid_feature2=centers[cluster_num][1],
                cluster_density=avg_distance,
                customer_ids=cluster_data['CustomerID'].tolist()
            )

        return analysis

    def handle(self, *args, **kwargs):
        # Define numeric fields
        numeric_fields = [
            'Age',
            'Gender',
            'Region',
            'City',
            'Feedback',
            'Income_Level',
        ]

        # Load data
        data = NumericData.objects.all().values('CustomerID', *numeric_fields)
        df = pd.DataFrame(data)

        # Create plots directory
        plot_dir = 'dataupload/plots'
        os.makedirs(plot_dir, exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create scaler
        scaler = StandardScaler()

        # Generate all possible pairs of features
        feature_pairs = list(itertools.combinations(numeric_fields, 2))

        # Process each pair
        for feature1, feature2 in feature_pairs:
            self.stdout.write(f'\nProcessing clustering for {feature1} vs {feature2}...')

            analysis = self.create_clustering_analysis(
                df,
                feature1,
                feature2,
                scaler,
                timestamp,
                plot_dir
            )

            if analysis:
                self.stdout.write(self.style.SUCCESS(
                    f'Analysis completed and stored. ID: {analysis.id}'
                ))

                # Print summary of stored clusters
                for cluster in analysis.clusters.all():
                    self.stdout.write(
                        f'\nCluster {cluster.cluster_number}:'
                        f'\n- Samples: {cluster.sample_count}'
                        f'\n- {feature1} mean: {cluster.feature1_mean:.2f}'
                        f'\n- {feature2} mean: {cluster.feature2_mean:.2f}'
                        f'\n- Density: {cluster.cluster_density:.3f}'
                    )

        self.stdout.write(self.style.SUCCESS('\nAll clustering analyses completed and stored!'))