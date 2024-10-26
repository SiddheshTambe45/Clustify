# dataupload/forms.py
from django import forms

class JSONUploadForm(forms.Form):
    file = forms.FileField()
