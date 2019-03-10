# -*- coding: utf-8 -*-
"""
Created on Sun Mar 03 17:55:26 2019
@author: nanda
"""
import numpy as np
from numpy import array
import csv
import random
from functools import reduce
from array import *
#GENERATE WEIGHT
def generateWeight(hidden_neuron, n_neuron, n):
    weight = []
    for row in range(hidden_neuron):
        x = random.random()*random.choice([-1*n,n])# random.random = nilai antara 0-1, random.choice([-1,1])=memilih antara -1 atau 1
        y = []
        for item in range(n_neuron):
            y.append(x)
        weight.append(y)
    print('Weight : ',weight)
    return weight
#NORMALISASI DATA
def normalisasi(alldata):
    maks = np.float64(max(alldata))
    mins = np.float64(min(alldata))
    norm = [list(map(lambda x: ((np.float64(x)-mins)/(maks-mins)), alldata))]
    return norm,maks,mins
#DENORMALISASI
def denormalisasi(alldata,maks,mins):
    unorm = [list(map(lambda x: ((np.float64(x)*(maks-mins))+mins), alldata))]
    return unorm
#LOAD HARGA CABAI
def loadData():
    alldata = []
    with open('hargacabai.csv', 'r') as csvFile:
        dt = csv.reader(csvFile, delimiter=';')
        for row in dt:
            for item in row:
                alldata.append(item)
    return alldata
#INISIALISASI
def inisialisasi(h_training, t_training, h_testing, t_testing, n_neuron, hidden_neuron, n):
    alldata = loadData()
    alldata,maks,mins = normalisasi(alldata)
    alldata = alldata[0]
    all_neuron = []
    target = []
    # Memetakan Semua Neuron Ke Target
    for i in range(len(alldata)-n_neuron):
        neuron = []
        for j in range(i,i+n_neuron):
            neuron.append(alldata[j])
            all_neuron.append(neuron)
            target.append(alldata[i+n_neuron])
    weight = generateWeight(hidden_neuron,n_neuron,n)
    training_data = [all_neuron[i] for i in range(h_training, t_training)]
    training_target = [target[i] for i in range(h_training, t_training)]
    testing_data = [all_neuron[i] for i in range(h_testing, t_testing)]
    testing_target = [target[i] for i in range(h_testing, t_testing)]
    return weight, training_data, training_target, testing_data, testing_target, maks, mins
def aktivasi(fungsi, x):
    if fungsi == 'sigmoid':
        result = 1/(1+np.exp(-1*x))
    else:
        result = 1/(1+np.exp(-1*x))
    return result
def learning(weight, training_data, training_target):
    #LEARNING
    Hinit = np.matmul(training_data, np.transpose(weight))
    H = [list(map(aktivasi, 'sigmoid', x)) for x in Hinit]
    #print('H : ',H)
    #print('HTH : ',np.matmul(np.transpose(H),H))
    Hdagger = np.matmul(np.linalg.inv(np.matmul(np.transpose(H),H)),np.transpose(H))
    B = np.matmul(Hdagger,training_target)
    return B
def hitungMAPE(prediksi, testing_target):
    error = list(np.abs((np.float64(a)-np.float64(f))/np.float64(a)) for a,f in zip(testing_target,prediksi))[0]
    sum = reduce(lambda x,y: x+y,error)
    MAPE = (sum/len(testing_target[0]))*100
    """
    print('Prediksi\t\tActual\t\tError')
    for i,j,k in zip(testing_target[0],prediksi[0],error):
        print(j,'\t',i,'\t',(k*100),'%')
    print(MAPE)
    """
    return MAPE,error
def testing(weight, maks, mins, beta, testing_data, testing_target):
    #TESTING
    Hinit = np.matmul(testing_data, np.transpose(weight))
    H = [list(map(aktivasi, 'sigmoid', x)) for x in Hinit]
    prediksi = np.matmul(H,beta)
    prediksi = denormalisasi(prediksi,maks,mins)
    target = denormalisasi(testing_target,maks,mins)
    MAPE,ERROR = hitungMAPE(prediksi,target)
    return prediksi,target,MAPE,ERROR
def elm(neuronS,neuronT,hiddenS,hiddenT):
    optimum = []
    for i in range(neuronS,neuronT):
        for j in range(hiddenS,hiddenT):
            print("============| Neuron : ",i," | Hidden Neuron : ",j," |============")
            result = []
            #for n in ([0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3]):
            for n in ([1]):
                for k in range(0,100):
                    weight, training_data, training_target, testing_data, testing_target, maks, mins = inisialisasi(0,81,81,111,i,j,n) 
                    beta = learning(weight,training_data,training_target)
                    prediksi,target,MAPE,ERROR = testing(weight,maks,mins,beta,testing_data,testing_target)
                    #print("MAPE : ",np.float64(MAPE))
                    result.append(np.float64(MAPE))
                    optimum.append(np.float64(MAPE))
            print("Minimum MAPE : ",np.min(result),"%")
            #print('==============================================================')
    print("OPTIMUM : ",np.min(optimum))
elm(9,10,3,4)