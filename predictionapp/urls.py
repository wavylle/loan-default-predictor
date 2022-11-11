from django.urls import path
from . import views

urlpatterns = [
    path('predictionpost', views.predictionpost, name='predictionpost'),
]
