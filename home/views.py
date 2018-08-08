from django.views.generic import TemplateView
from home.forms import HomeForm
from home.models import Post
from django.shortcuts import render,redirect
from home.beliefEng.end2end import tag
from django.conf import settings


class Homeview(TemplateView):
    template_name = 'home/home.html'
    pFlag=False
    def get(self,request):
        form=HomeForm()
        args = {'form': form}
        return render(request, self.template_name,args)
    def post(self,request):
        form =HomeForm(request.POST)
        if form.is_valid():
            post=form.save(commit=False)
            post.user=request.user
            post.save()
            self.pFlag=True
            model = settings.MODEL
            tkz=settings.TOKENIZER
            text=form.cleaned_data['post']
            form=HomeForm()
            tagged = tag(model,tkz(post))
            args = {'form': form, 'posts': post, 'tagged': tagged}
            return render(request,self.template_name,args)

        args={'form':form,'text':text}
        return render(request,self.template_name,args)
