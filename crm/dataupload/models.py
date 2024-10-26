from django.db import models

# Create your models here.

from django.db import models

# class JSONData(models.Model):
#     # This model will store JSON data dynamically
#     name = models.CharField(max_length=100,default='unknown')
#     age = models.IntegerField(default=None)

class JSONData(models.Model):
    # This model will store customer-related JSON data dynamically
    CustomerID = models.CharField(max_length=50, unique=False, blank=False, default='unknown')
    Name = models.CharField(max_length=100, blank=False, default='unknown')
    Age = models.IntegerField(blank=True, null=True)  # Allows blank and null values
    Gender = models.CharField(max_length=10, blank=True, default='Not specified')  # Consider using choices for better validation
    Region = models.CharField(max_length=100, blank=True, default='Not specified')
    City = models.CharField(max_length=100, blank=True, default='Not specified')
    Interests = models.TextField(blank=True, default='')  # Allows blank but has a default empty string
    Purchase_History = models.JSONField(blank=True, default=dict)  # Default to an empty dictionary
    Last_Purchase_Date = models.DateField(blank=True, null=True)  # Allows blank and null values
    Feedback = models.TextField(blank=True, default='')  # Allows blank but has a default empty string
    Income_Level = models.CharField(max_length=50, blank=True, default='Not specified')

    def __str__(self):
        return f"{self.Name} ({self.CustomerID})"


class NumericData(models.Model):
    CustomerID = models.CharField(max_length=50, unique=True, blank=False, default='unknown')
    Age = models.IntegerField(blank=True, null=True)  # Store age as numeric
    Gender = models.IntegerField(blank=True, null=True)  # Encode Gender as numeric (e.g., Male=1, Female=0)
    Region = models.IntegerField(blank=True, null=True)  # Encode regions as numeric
    City = models.IntegerField(blank=True, null=True)  # Encode cities as numeric
    Interests = models.JSONField(blank=True, default=dict)  # Keep original interests for reference
    Feedback = models.IntegerField(blank=True, null=True)  # Convert feedback to numeric
    Income_Level = models.IntegerField(blank=True, null=True)  # Encode income levels as numeric
    Purchase_History = models.JSONField(blank=True, default=dict)

    def __str__(self):
        return f"Numeric Data for {self.CustomerID}"


class ClusterAnalysis(models.Model):
    # Basic cluster information
    analysis_date = models.DateTimeField(auto_now_add=True)
    feature1 = models.CharField(max_length=50)
    feature2 = models.CharField(max_length=50)
    plot_path = models.CharField(max_length=255)

    # Cluster metrics
    total_samples = models.IntegerField()
    number_of_clusters = models.IntegerField(default=3)
    silhouette_score = models.FloatField(null=True)  # Overall clustering quality

    def __str__(self):
        return f"Cluster Analysis: {self.feature1} vs {self.feature2} ({self.analysis_date})"


class ClusterDetails(models.Model):
    # Link to main analysis
    analysis = models.ForeignKey(ClusterAnalysis, on_delete=models.CASCADE, related_name='clusters')
    cluster_number = models.IntegerField()

    # Cluster statistics
    sample_count = models.IntegerField()
    feature1_mean = models.FloatField()
    feature1_std = models.FloatField()
    feature2_mean = models.FloatField()
    feature2_std = models.FloatField()

    # Centroid location
    centroid_feature1 = models.FloatField()
    centroid_feature2 = models.FloatField()

    # Additional metrics
    cluster_density = models.FloatField(null=True)  # Average distance to centroid

    # Store sample IDs for this cluster
    customer_ids = models.JSONField(default=list)  # Store CustomerIDs in this cluster

    class Meta:
        unique_together = ('analysis', 'cluster_number')

    def __str__(self):
        return f"Cluster {self.cluster_number} of {self.analysis}"


class Image(models.Model):
    title = models.CharField(max_length=200)
    image = models.ImageField(upload_to='media/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
