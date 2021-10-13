# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 07:41:57 2021

@author: janhe
"""


import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join


methoden = ["Ergebnisse_ARIMA", "Ergebnisse_LSTM", "Ergebnisse_CNN"]
columns = ['Aktie', 'aktueller_kurs', 'vorhersage', 'prozentuale Änderung', 'tatsächlich', 'absolute abweichung', 'realtive abweichung', 'datum']

statistische_Auswertung = pd.DataFrame(columns = ["METHODE", "RMSE", "MAE", "MIN", "MAX", "MAPE", "sMAPE", "MdAPE"], dtype=object)



for methode in methoden:
    datums_liste = os.listdir(os.path.join(r"D:\Statistik Masterarbeit\Daten", methode ).replace("\\","/"))
    auswertung = pd.DataFrame(dtype=object)
    for i in range(0, len(datums_liste)):
            df = pd.read_csv(os.path.join(r"D:\Statistik Masterarbeit\Daten", methode, datums_liste[i]).replace("\\","/"), encoding='latin-1')
            if sum(df.columns == "Unnamed: 0") > 0:
                df = df.drop("Unnamed: 0", axis=1)
            auswertung = auswertung.append(df)
    if methode == "Ergebnisse_ARIMA":      
        daten = auswertung[['Prognose', 'tatsächlich']].dropna()
    else:
        daten = auswertung[['vorhersage', 'tatsÃ¤chlich']].dropna()
        daten = daten.rename(columns = {"vorhersage": "Prognose", "tatsÃ¤chlich" : "tatsächlich"})
    
    RMSE = np.square(np.subtract(daten["tatsächlich"], daten['Prognose'])).mean()
    MAE = abs(np.subtract(daten["tatsächlich"], daten['Prognose'])).mean()
    MIN = np.subtract(daten["tatsächlich"], daten['Prognose']).min()
    MAX = np.subtract(daten["tatsächlich"], daten['Prognose']).max()
    MAPE = np.mean(np.abs(np.subtract(daten["tatsächlich"], daten['Prognose']) / daten["tatsächlich"])) * 100
    MdAPE = np.median(np.abs(np.subtract(daten["tatsächlich"], daten['Prognose']) / (daten["tatsächlich"]+daten["Prognose"]))) * 100 
    sMAPE = 100/len(daten) * np.sum(2 * np.abs(daten['Prognose'] - daten["tatsächlich"]) / (np.abs(daten["tatsächlich"]) + np.abs(daten['Prognose'])))
    METHODE = methode
    statistische_Auswertung.loc[len(statistische_Auswertung)] = [METHODE, RMSE, MAE, MIN, MAX, MAPE, sMAPE, MdAPE]
        
    
(daten["tatsächlich"] -daten["Prognose"]).max()
(daten["tatsächlich"] -daten["Prognose"]).min()
daten



#Auswetung pre corona vs post

#pre

methoden = ["Ergebnisse_ARIMA", "Ergebnisse_LSTM", "Ergebnisse_CNN"]
columns = ['Aktie', 'aktueller_kurs', 'vorhersage', 'prozentuale Änderung', 'tatsächlich', 'absolute abweichung', 'realtive abweichung', 'datum']

statistische_Auswertung = pd.DataFrame(columns = ["METHODE", "MSE", "MAE", "MIN", "MAX", "MAPE", "sMAPE", "MdAPE"], dtype=object)




for methode in methoden:
    datums_liste = os.listdir(os.path.join(r"D:\Statistik Masterarbeit\Daten", methode ).replace("\\","/"))
    auswertung = pd.DataFrame(dtype=object)
    for i in range(0, datums_liste.index("2019-12-30")):
            df = pd.read_csv(os.path.join(r"D:\Statistik Masterarbeit\Daten", methode, datums_liste[i]).replace("\\","/"), encoding='latin-1')
            if sum(df.columns == "Unnamed: 0") > 0:
                df = df.drop("Unnamed: 0", axis=1)
            auswertung = auswertung.append(df)
    if methode == "Ergebnisse_ARIMA":      
        daten = auswertung[['Prognose', 'tatsächlich']].dropna()
    else:
        daten = auswertung[['vorhersage', 'tatsÃ¤chlich']].dropna()
        daten = daten.rename(columns = {"vorhersage": "Prognose", "tatsÃ¤chlich" : "tatsächlich"})
    
    MSE = np.square(np.subtract(daten["tatsächlich"], daten['Prognose'])).mean()
    MAE = abs(np.subtract(daten["tatsächlich"], daten['Prognose'])).mean()
    MIN = np.subtract(daten["tatsächlich"], daten['Prognose']).min()
    MAX = np.subtract(daten["tatsächlich"], daten['Prognose']).max()
    MAPE = np.mean(np.abs(np.subtract(daten["tatsächlich"], daten['Prognose']) / daten["tatsächlich"])) * 100
    MdAPE = np.median(np.abs(np.subtract(daten["tatsächlich"], daten['Prognose']) / (daten["tatsächlich"]+daten["Prognose"]))) * 100 
    sMAPE = 100/len(daten) * np.sum(2 * np.abs(daten['Prognose'] - daten["tatsächlich"]) / (np.abs(daten["tatsächlich"]) + np.abs(daten['Prognose'])))
    METHODE = methode
    statistische_Auswertung.loc[len(statistische_Auswertung)] = [METHODE, MSE, MAE, MIN, MAX, MAPE, sMAPE, MdAPE]
        
print(statistische_Auswertung)
    
#post


methoden = ["Ergebnisse_ARIMA", "Ergebnisse_LSTM", "Ergebnisse_CNN"]
columns = ['Aktie', 'aktueller_kurs', 'vorhersage', 'prozentuale Änderung', 'tatsächlich', 'absolute abweichung', 'realtive abweichung', 'datum']

statistische_Auswertung = pd.DataFrame(columns = ["METHODE", "MSE", "MAE", "MIN", "MAX", "MAPE", "sMAPE", "MdAPE"], dtype=object)




for methode in methoden:
    datums_liste = os.listdir(os.path.join(r"D:\Statistik Masterarbeit\Daten", methode ).replace("\\","/"))
    auswertung = pd.DataFrame(dtype=object)
    for i in range(datums_liste.index("2019-12-30"), len(datums_liste)):
            df = pd.read_csv(os.path.join(r"D:\Statistik Masterarbeit\Daten", methode, datums_liste[i]).replace("\\","/"), encoding='latin-1')
            if sum(df.columns == "Unnamed: 0") > 0:
                df = df.drop("Unnamed: 0", axis=1)
            auswertung = auswertung.append(df)
    if methode == "Ergebnisse_ARIMA":      
        daten = auswertung[['Prognose', 'tatsächlich']].dropna()
    else:
        daten = auswertung[['vorhersage', 'tatsÃ¤chlich']].dropna()
        daten = daten.rename(columns = {"vorhersage": "Prognose", "tatsÃ¤chlich" : "tatsächlich"})
    
    MSE = np.square(np.subtract(daten["tatsächlich"], daten['Prognose'])).mean()
    MAE = abs(np.subtract(daten["tatsächlich"], daten['Prognose'])).mean()
    MIN = np.subtract(daten["tatsächlich"], daten['Prognose']).min()
    MAX = np.subtract(daten["tatsächlich"], daten['Prognose']).max()
    MAPE = np.mean(np.abs(np.subtract(daten["tatsächlich"], daten['Prognose']) / daten["tatsächlich"])) * 100
    MdAPE = np.median(np.abs(np.subtract(daten["tatsächlich"], daten['Prognose']) / (daten["tatsächlich"]+daten["Prognose"]))) * 100 
    sMAPE = 100/len(daten) * np.sum(2 * np.abs(daten['Prognose'] - daten["tatsächlich"]) / (np.abs(daten["tatsächlich"]) + np.abs(daten['Prognose'])))
    METHODE = methode
    statistische_Auswertung.loc[len(statistische_Auswertung)] = [METHODE, MSE, MAE, MIN, MAX, MAPE, sMAPE, MdAPE]
        
print(statistische_Auswertung)
    
    
        
        
        
        auswertung.columns
        
        
    "D:\Statistik Masterarbeit\Daten\Ergebnisse_ARIMA\2017-12-30"
    pd.read_csv(os.path.join(r"D:\Statistik Masterarbeit\Daten", methode, datums_liste[1]).replace("\\","/"))
    
    
    print(methode)
    

performances = pd.DataFrame(columns: ['Datum', 'Performance', 'Anzahl'])

for loop über die Zeitpunkte
daten = pd.read_csv("/content/2017-12-30", encoding='latin-1')

erwartete_Veränderung = daten['Prognose']/daten['aktueller Kurs']-1

neu = daten[['Aktie', 'aktueller Kurs', 'Prognose', 'tatsächlich']].copy()

neu['erwartete_Veränderung'] = neu['Prognose']/neu['aktueller Kurs']-1
neu['tatsächliche_Veränderung'] = neu['tatsächlich']/neu['aktueller Kurs']-1
neu = neu.sort_values(by='erwartete_Veränderung', ascending=False)
performance = neu[:5]['tatsächliche_Veränderung'].mean()





##### Auswertung statistisch monatlich

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 07:41:57 2021

@author: janhe
"""


import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

methoden = ["Ergebnisse_ARIMA", "Ergebnisse_LSTM", "Ergebnisse_CNN"]
columns = ['Aktie', 'aktueller_kurs', 'vorhersage', 'prozentuale Änderung', 'tatsächlich', 'absolute abweichung', 'realtive abweichung', 'datum']

Monatsauswertung = {}

for methode in methoden:
    datums_liste = os.listdir(os.path.join(r"D:\Statistik Masterarbeit\Daten", methode ).replace("\\","/"))
    auswertung = pd.DataFrame(dtype=object)
    statistische_Auswertung = pd.DataFrame(columns = ["METHODE", "RMSE", "MAE", "MIN", "MAX", "MAPE", "sMAPE"], dtype=object)

    for i in range(0, len(datums_liste)):
        df = pd.read_csv(os.path.join(r"D:\Statistik Masterarbeit\Daten", methode, datums_liste[i]).replace("\\","/"), encoding='latin-1')
        if methode == "Ergebnisse_ARIMA":      
            daten = df[['Prognose', 'tatsächlich']].dropna()
        else:
            daten = df[['vorhersage', 'tatsÃ¤chlich']].dropna()
            daten = daten.rename(columns = {"vorhersage": "Prognose", "tatsÃ¤chlich" : "tatsächlich"})
            
        MSE = np.square(np.subtract(daten["tatsächlich"], daten['Prognose'])).mean()
        MAE = abs(np.subtract(daten["tatsächlich"], daten['Prognose'])).mean()
        MIN = np.subtract(daten["tatsächlich"], daten['Prognose']).min()
        MAX = np.subtract(daten["tatsächlich"], daten['Prognose']).max()
        MAPE = np.mean(np.abs(np.subtract(daten["tatsächlich"], daten['Prognose']) / daten["tatsächlich"])) * 100
        sMAPE = 100/len(daten) * np.sum(2 * np.abs(daten['Prognose'] - daten["tatsächlich"]) / (np.abs(daten["tatsächlich"]) + np.abs(daten['Prognose'])))
        METHODE = methode
        statistische_Auswertung.loc[len(statistische_Auswertung)] = [METHODE, MSE, MAE, MIN, MAX, MAPE, sMAPE]
    statistische_Auswertung.index=datums_liste
    Monatsauswertung[methode] = statistische_Auswertung
        

Kennzahlen = ["RMSE", "MAE", "MIN", "MAX", "MAPE", "sMAPE"]

for kennzahl in Kennzahlen:
    Monatsauswertung["Ergebnisse_ARIMA"][kennzahl].plot(grid=True, label="ARIMA")
    Monatsauswertung["Ergebnisse_LSTM"][kennzahl].plot(grid=True, label="LSTM")
    Monatsauswertung["Ergebnisse_CNN"][kennzahl].plot(grid=True, label="CNN")
    plt.xticks(rotation=45)
    plt.legend()
    plt.xlabel('Datum')
    plt.ylabel(str(kennzahl))
    plt.gcf().subplots_adjust(bottom=0.35)
    plt.savefig(os.path.join(r"D:\Statistik Masterarbeit\Daten\Bilder_Auswertung_monatlich", str(kennzahl)).replace("\\","/"), encoding='latin-1', dpi=1200)
    plt.show()
    
    
for methode in methoden:
    datums_liste = os.listdir(os.path.join(r"D:\Statistik Masterarbeit\Daten", methode ).replace("\\","/"))
    auswertung = pd.DataFrame(dtype=object)
    statistische_Auswertung = pd.DataFrame(columns = ["METHODE", "MSE", "MAE", "MIN", "MAX", "MAPE", "sMAPE"], dtype=object)
    counter = 0
    for i in range(0, len(datums_liste)):
        df = pd.read_csv(os.path.join(r"D:\Statistik Masterarbeit\Daten", methode, datums_liste[i]).replace("\\","/"), encoding='latin-1')
        if methode == "Ergebnisse_ARIMA":      
            daten = df[['Prognose', 'tatsächlich']].dropna()
        else:
            daten = df[['vorhersage', 'tatsÃ¤chlich']].dropna()
            daten = daten.rename(columns = {"vorhersage": "Prognose", "tatsÃ¤chlich" : "tatsächlich"})
    
        counter = counter + sum(abs(np.subtract(daten["tatsächlich"], daten['Prognose'])) > 50)
    print(counter)
    
    
    
    
D:\Statistik Masterarbeit\Daten\Bilder_Auswertung_monatlich

os.path.join(r"D:\Statistik Masterarbeit\Bilder_Auswertung_monatlich", kennzahl).replace("\\","/")


df.set_index(pd.to_datetime(df.date), drop=True).plot()



import matplotlib.pyplot as plt
plt.plot(Monatsauswertung["Ergebnisse_ARIMA"]["MSE"])
plt.plot(Monatsauswertung["Ergebnisse_LSTM"]["MSE"])

        
        
        
        auswertung.columns
        
        
    "D:\Statistik Masterarbeit\Daten\Ergebnisse_ARIMA\2017-12-30"
    pd.read_csv(os.path.join(r"D:\Statistik Masterarbeit\Daten", methode, datums_liste[1]).replace("\\","/"))
    
    
    print(methode)
    

performances = pd.DataFrame(columns: ['Datum', 'Performance', 'Anzahl'])

for loop über die Zeitpunkte
daten = pd.read_csv("/content/2017-12-30", encoding='latin-1')

erwartete_Veränderung = daten['Prognose']/daten['aktueller Kurs']-1

neu = daten[['Aktie', 'aktueller Kurs', 'Prognose', 'tatsächlich']].copy()

neu['erwartete_Veränderung'] = neu['Prognose']/neu['aktueller Kurs']-1
neu['tatsächliche_Veränderung'] = neu['tatsächlich']/neu['aktueller Kurs']-1
neu = neu.sort_values(by='erwartete_Veränderung', ascending=False)
performance = neu[:5]['tatsächliche_Veränderung'].mean()
h
