#Backtest der Handelsstrategien auf Basis der Prognosen

#Importieren der benötigten Bibliotheken
import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt


methoden = ["Ergebnisse_ARIMA", "Ergebnisse_LSTM", "Ergebnisse_CNN"] #definiert eine Liste mit den verwendeten Methoden

erg =  {} #erstellten ein leeres Dictionary

for methode in methoden: #loop über Methoden
    a = 0
    performances = [a]
    datums_liste = os.listdir(os.path.join(r"D:\Statistik Masterarbeit\Daten", methode ).replace("\\","/")) #liest die Liste mit den Datumsangaben ein
    datums_liste.sort() #sortiert die Liste mit den Datumsangaben
    auswertung = pd.DataFrame(dtype=object) #erstellt eine leere Tabelle für die Auswertung
    for datum in datums_liste: #Schleife über die Datumsangaben
            df = pd.read_csv(os.path.join(r"D:\Statistik Masterarbeit\Daten", methode, datum).replace("\\","/"), encoding='latin-1') #liest die Prognosen ein
            if methode == "Ergebnisse_ARIMA":  #bearbeitet die Daten je nachdem mit welcher Methode die Prognosen erstellt wurden (Hintergrund ist der, dass die CSV jeweils etwas anders gestaltet ist)
                daten = df[['Aktie', 'aktueller Kurs', 'Prognose', 'tatsächlich']].dropna()
                daten = daten.rename(columns = {'aktueller Kurs': 'aktuell'})
                df.columns
            else:
                daten = df[['Aktie', 'aktueller_kurs', 'vorhersage', 'tatsÃ¤chlich']].dropna()
                daten = daten.rename(columns = {'aktueller_kurs': "aktuell" ,"vorhersage": "Prognose", "tatsÃ¤chlich" : "tatsächlich"})
            daten['erwartete Rendite'] = daten['Prognose']/daten['aktuell']-1 #berechnet die erwartete Rendite auf Basis der Prognosen
            daten['tatsächliche Rendite']= daten['tatsächlich']/daten['aktuell']-1 #berechnet die tatsächlich Rendite 
            daten = daten.sort_values('erwartete Rendite', ascending = False) #sortiert die Liste absteigend nach den erwarteten Renditen
            performance_long = list(daten[:5]['tatsächliche Rendite']) #die Rendite der besten Fünf
            performance_short = list(-daten[-5:]['tatsächliche Rendite']) #und die negative Rendite (Short-Position) der schlechtesten fünf Prognosen wird gewählt
            renditen = performance_long + performance_short #beide Renditen werden aufaddiert
            performance = np.mean(renditen) #Gesamtperformance ist Mittelwert der einzelnen Renditen
            performances.append(performance) #die Rendite in diesem Monat wird an den DataFrame anhängt
            output = pd.DataFrame(list(zip(performances)), columns = ["Datum", "Performance"]) #Die liste wird für den Output vorbereitet      
            output["Performance kumuliert"] = (1+output['Performance']).cumprod() #die kumulierte Performance wird errechnet
    output.to_csv(os.path.join(r"D:\Statistik Masterarbeit\Daten\Trading\5_long_5_short", methode).replace("\\","/")) #speichert die Werte in einer CSCV
    erg[methode] = (output["Performance kumuliert"]-1) #die Kapitalkurve wird in das Dictionary eingefügt

    
    
    #Berechnung der Risikokennzahlen gemäß der Formeln in der Arbeit
    #Sharpe Ratio
    SR = output['Performance'].mean()/output['Performance'].std()
    #Sortino Ratio
    df = pd.DataFrame()
    df['Returns'] = output['Performance']
    df['downside_returns'] = 0
    df.loc[df['Returns'] < 0, 'downside_returns'] = df['Returns']**2
    expected_return = df['Returns'].mean()
    down_stdev = np.sqrt(df['downside_returns'].mean())
    sortino_ratio = (expected_return - 0)/down_stdev
    #Maximum Drawdown
    MD = abs((((output['Performance']+1).cumprod()-(output['Performance']+1).cumprod().cummax())/(output['Performance']+1).cumprod().cummax())).max()
    #Speichern der Risikokennzahlen in einer .csv Datei 
    risikokennzahlen = pd.DataFrame([SR, sortino_ratio, MD],)
    risikokennzahlen.index = ["SR", "sortino_ratio", "MD"]
    risikokennzahlen.to_csv(os.path.join(r"D:\Statistik Masterarbeit\Daten\Trading\5_long_5_short", str(methode + "Risiko")).replace("\\","/"))
            
