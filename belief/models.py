from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save

# Create your models here.
#this is the scheme of the dataset


class UserProfile(models.Model):
    user = models.OneToOneField(User,on_delete=models.CASCADE)
    description = models.CharField(max_length=100,default='')
    city=models.CharField(max_length=50,default='')
    website=models.URLField(default='')
    phone =models.IntegerField(default=0)


