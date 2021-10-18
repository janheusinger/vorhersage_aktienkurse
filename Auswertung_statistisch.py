#Auswertung der Prognosen mit Hilfe statistischer Kennzahlen


#Importieren der benötigten Bibliotheken
import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join


methoden = ["Ergebnisse_ARIMA", "Ergebnisse_LSTM", "Ergebnisse_CNN"] #definiert Liste mit den verwendeten Methoden
columns = ['Aktie', 'aktueller_kurs', 'vorhersage', 'prozentuale Änderung', 'tatsächlich', 'absolute abweichung', 'realtive abweichung', 'datum'] #definiert die Liste mit den verwendeten Kennzahlen

statistische_Auswertung = pd.DataFrame(columns = ["METHODE", "RMSE", "MAE", "MIN", "MAX", "MAPE", "sMAPE", "MdAPE"], dtype=object) #erstellt die Tabelle für die Auswertung

for methode in methoden: #Schleife über die Mehtoden 
    datums_liste = os.listdir(os.path.join(r"D:\Statistik Masterarbeit\Daten", methode ).replace("\\","/")) #einlesen der Datumslist
    auswertung = pd.DataFrame(dtype=object) #erstellt leeren DataFrame für Auswertung
    for i in range(0, len(datums_liste)): #Schleife über die Datumsangaben
            df = pd.read_csv(os.path.join(r"D:\Statistik Masterarbeit\Daten", methode, datums_liste[i]).replace("\\","/"), encoding='latin-1') #List die Prognosen der aktuellen Methode ein
            if sum(df.columns == "Unnamed: 0") > 0: #Verarbeitung der Daten, da die CSV-Dateien für ARIMA-GACH und ML Prognosen anders aussehen
                df = df.drop("Unnamed: 0", axis=1)
            auswertung = auswertung.append(df)
    if methode == "Ergebnisse_ARIMA":      
        daten = auswertung[['Prognose', 'tatsächlich']].dropna()
    else:
        daten = auswertung[['vorhersage', 'tatsÃ¤chlich']].dropna()
        daten = daten.rename(columns = {"vorhersage": "Prognose", "tatsÃ¤chlich" : "tatsächlich"})
    #Berechnung der verschiedenen Kennzahlen
    RMSE = np.square(np.subtract(daten["tatsächlich"], daten['Prognose'])).mean()
    MAE = abs(np.subtract(daten["tatsächlich"], daten['Prognose'])).mean()
    MIN = np.subtract(daten["tatsächlich"], daten['Prognose']).min()
    MAX = np.subtract(daten["tatsächlich"], daten['Prognose']).max()
    MAPE = np.mean(np.abs(np.subtract(daten["tatsächlich"], daten['Prognose']) / daten["tatsächlich"])) * 100
    MdAPE = np.median(np.abs(np.subtract(daten["tatsächlich"], daten['Prognose']) / (daten["tatsächlich"]+daten["Prognose"]))) * 100 
    sMAPE = 100/len(daten) * np.sum(2 * np.abs(daten['Prognose'] - daten["tatsächlich"]) / (np.abs(daten["tatsächlich"]) + np.abs(daten['Prognose'])))
    METHODE = methode
    statistische_Auswertung.loc[len(statistische_Auswertung)] = [METHODE, RMSE, MAE, MIN, MAX, MAPE, sMAPE, MdAPE] #einfügen der Ergebnisse in die Ergebnistabelle

#Auswetung pre corona vs post

#pre, funktioniert wie oben, wobei jedoch nur Daten bis Dezember 2019 berücksichtigt werden
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
    
#post, funktioniert wie oben, wobei jedoch nur Daten ab Januar 2020 berücksichtigt werden
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
     
        
     

#percentage better

import itertools
for methode, methode2 in itertools.combinations(methoden, 2):
            
            datums_liste = os.listdir(os.path.join(r"D:\Statistik Masterarbeit\Daten", methode ).replace("\\","/"))
            auswertung = pd.DataFrame(dtype=object)
            zähler = 0
            
            for i in range(datums_liste.index("2019-12-30"), len(datums_liste)):
                    df = pd.read_csv(os.path.join(r"D:\Statistik Masterarbeit\Daten", methode, datums_liste[i]).replace("\\","/"), encoding='latin-1')
                    if sum(df.columns == "Unnamed: 0") > 0:
                        df = df.drop("Unnamed: 0", axis=1)
                    if methode == "Ergebnisse_ARIMA":      
                            df = df[['Prognose', 'tatsächlich', 'Datum']].dropna()
                            df = df.set_index(df["Datum"])
                    else:
                            df = df[['vorhersage', 'tatsÃ¤chlich', 'datum']].dropna()
                            df = df.rename(columns = {"vorhersage": "Prognose", "tatsÃ¤chlich" : "tatsächlich"})
                            df = df.set_index(df["datum"])
                    MAPE1 = np.mean(np.abs(np.subtract(df["tatsächlich"], df['Prognose']) / df["tatsächlich"])) * 100
                    df = pd.read_csv(os.path.join(r"D:\Statistik Masterarbeit\Daten", methode2, datums_liste[i]).replace("\\","/"), encoding='latin-1')
                    if sum(df.columns == "Unnamed: 0") > 0:
                        df = df.drop("Unnamed: 0", axis=1)
                    if methode2 == "Ergebnisse_ARIMA":      
                            df = df[['Prognose', 'tatsächlich', 'Datum']].dropna()
                            df = df.set_index(df["Datum"])
                    else:
                            df = df[['vorhersage', 'tatsÃ¤chlich', 'datum']].dropna()
                            df = df.rename(columns = {"vorhersage": "Prognose", "tatsÃ¤chlich" : "tatsächlich"})
                            df = df.set_index(df["datum"])
                    MAPE2 = np.mean(np.abs(np.subtract(df["tatsächlich"], df['Prognose']) / df["tatsächlich"])) * 100
                    if MAPE1>MAPE2:
                        zähler = zähler +1
                    else:
                        zähler = zähler
            percentage_better = zähler/len(datums_liste)
            print(zähler, methode, methode2, percentage_better)








