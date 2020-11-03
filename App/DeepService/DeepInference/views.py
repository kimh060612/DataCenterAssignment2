from django.shortcuts import render
from django.http import HttpResponse
import requests

# Create your views here.
def say_hello(request):
    return HttpResponse("<h1>Hello Fucking World!</h1>")

def predict():
    pass