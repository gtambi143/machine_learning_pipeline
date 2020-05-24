from django.urls import path
from classify import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload_image', views.upload_image, name="upload_image"),
    path('upload_training_sample', views.store_training_set, name="upload_training_sample")
]
