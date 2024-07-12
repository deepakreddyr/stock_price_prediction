from django.urls import path

from . import views

urlpatterns = [
    
    path("", views.home, name="home"),
    path('output/', views.my_view, name='my_view'),
]