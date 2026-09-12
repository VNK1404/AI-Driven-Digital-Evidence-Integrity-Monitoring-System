from django.urls import path
from . import views

app_name = 'ml_app'

urlpatterns = [
    path('', views.index, name='home'),
    path('predict/', views.predict_page, name='predict'),
    path('about/', views.about, name='about'),
]
