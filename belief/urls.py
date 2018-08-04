from django.contrib.auth import views as auth_views
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('login/', auth_views.LoginView.as_view(template_name='belief/login.html')),
]