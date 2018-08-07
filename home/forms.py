from django import forms
from home.models import Post

class HomeForm(forms.ModelForm):
    post=forms.CharField(widget=forms.TextInput(
        attrs={
            'style':'height: 150px',
            'class':'form-control form-control-lg',
            'placeholder':'Write a document....',

    }))
    class Meta:
        model=Post
        fields=('post',)