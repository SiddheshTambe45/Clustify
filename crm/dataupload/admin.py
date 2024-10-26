from django.contrib import admin
from .models import JSONData,NumericData,ClusterDetails,ClusterAnalysis, Image
# Register your models here.

admin.site.register(JSONData)
admin.site.register(NumericData)
admin.site.register(ClusterDetails)
admin.site.register(ClusterAnalysis)
admin.site.register(Image)