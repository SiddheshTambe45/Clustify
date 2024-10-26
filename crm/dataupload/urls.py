from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_json, name='upload_json'),
    path('success/', views.success, name='success'),  # Create a success page
    path('records/', views.records_view, name='records'),
    # path('transformdata/', views.TransformDataView.as_view(), name='transformdata')
    path('transform/', views.transform_data, name='transformdata'),
    path('analysis/', views.analysis_view, name='analysis'),
    path('apianalysis/', views.analyze_cluster_with_chat, name='send_data_to_gemini'),
]
