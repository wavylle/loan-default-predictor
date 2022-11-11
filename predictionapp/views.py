from django.shortcuts import render
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.metrics import  precision_recall_curve, roc_auc_score, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score,auc, roc_curve, plot_confusion_matrix
import matplotlib.pyplot as plt
color = sns.color_palette()
import pingouin as pg
from sklearn.preprocessing import OneHotEncoder
from .prediction import predictDefault

def index(request):
    return render(request, 'index.html')

@csrf_exempt
def predictionpost(request):
    if request.method == 'POST':
        grade = request.POST['grade']
        annualinc = request.POST['annualinc']
        shortemp = request.POST['shortemp']
        emplength = request.POST['emplength']
        ownershiptype = request.POST['ownershiptype']
        dti = request.POST['dti']
        purpose = request.POST['purpose']
        term = request.POST['term']
        lastdelinqnone = request.POST['lastdelinqnone']
        revolutil = request.POST['revolutil']
        totalreclatefee = request.POST['totalreclatefee']
        odratio = request.POST['odratio']

        predictionResult = predictDefault(grade, annualinc, shortemp, emplength, ownershiptype, dti, purpose, term, lastdelinqnone, revolutil, totalreclatefee, odratio)

        return render(request, 'result.html', {'prediction_result': predictionResult})
        # return HttpResponse(predictionResult)
