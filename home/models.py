from django.db import models
from django.contrib.auth.models import User
#this is to store what people pot for tagging
# Create your models here.
class Post(models.Model):
    post=models.CharField(max_length=500)
    user=models.ForeignKey(User,on_delete=models.PROTECT)
    created=models.DateTimeField(auto_now_add=True)
    updated=models.DateTimeField(auto_now=True)