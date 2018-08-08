from django.views.generic import TemplateView
from home.forms import HomeForm
from home.models import Post
from django.shortcuts import render,redirect
from home.beliefEng.end2end import tag
from django.conf import settings


class Homeview(TemplateView):
    template_name = 'home/home.html'
    def get(self,request):
        model = settings.MODEL
        tkz=settings.TOKENIZER
        form=HomeForm()
        posts=Post.objects.all().order_by('-created')[:1]
        # if Post
        tagged=tag(model,tkz.tokenize(posts[0].post+"."))
        args={'form':form,'posts':posts,'tagged':tagged}
        return render(request,self.template_name,args)
    def post(self,request):
        form =HomeForm(request.POST)
        if form.is_valid():
            post=form.save(commit=False)
            post.user=request.user
            post.save()
            text=form.cleaned_data['post']
            form=HomeForm()
            return  redirect ('home:home')

        args={'form':form,'text':text}
        return render(request,self.template_name,args)

