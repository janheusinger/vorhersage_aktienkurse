#Code zur Vorhersage von Aktienkursen mit Hilfe von Long short-term memory Netzwerken

#importieren der notwendigend Bibliotheken
import numpy as np
import pandas as pd
from numpy import array
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.preprocessing import MinMaxScaler

# definieren der Funktion, um die Zeitreihe in Sequence zu Splitte, Quelle: Brownlee (2017)
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		if out_end_ix > len(sequence):
			break
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

#Einlesen und bearbeiten der Daten
dax_mitglieder = pd.read_csv(r"C:\Users\janhe\Documents\Masterarbeit Statistik\Daten\DAX_Liste_final.csv", sep=";", header=None)
dax_mitglieder[0] = pd.to_datetime(dax_mitglieder[0])
dax_mitglieder.set_index(0, inplace=True)


for i in range(0,len(dax_mitglieder)): #Schleife über die Datumsangaben in Liste dax_mitglieder
    aktuelle_mitglieder = list(dax_mitglieder.iloc[i]) #lädt die aktuellen Mitglieder des Index in die Variable
    aktuelles_datum = dax_mitglieder.index[i] #setzt das aktuelle Datum
    zwischenergebnis_matrix = pd.DataFrame(columns = ["Aktie", "aktueller_kurs", "vorhersage", "prozentuale_Änderung", "tatsächlich", "absolute_abweichung", "realtive_abweichung", "datum"], dtype=object) #initiert die Tabelle um Ergebnisse zu speichern

    for mitglied in aktuelle_mitglieder: #Schleife über die Mitglieder im aktuellen Monat
        data  = pd.read_csv(os.path.join(r"D:\Statistik Masterarbeit\Daten\Aktienkurse_adjustiert", mitglied+".csv.csv").replace("\\","/"))[["adjustierung", "Datum", "RIC"]] #lädt die Zeitreihe der aktuellen Aktie in die Varibaale
        data['Datum'] = pd.to_datetime(data['Datum']) #neue Spalte "Datum" wird erstellt
        data = data.sort_values('Datum', ascending=True) #sortieren der Werte
        #Der folgende Codeabschnitt dient zum Auffinden der Zeile mit dem aktuellen Monatsende
        hilfs_df = data
        hilfs_df['Monat'] = hilfs_df['Datum'].dt.month
        hilfs_df['Monat_shift'] = hilfs_df['Monat'].shift(-1)
        hilfs_df['Jahr'] = hilfs_df['Datum'].dt.year
        alle_monatsenden = hilfs_df[hilfs_df['Monat'] != hilfs_df['Monat_shift']]
        relevante_monatsenden = alle_monatsenden[(hilfs_df['Datum'] > "2017-12-01") & (hilfs_df['Datum'] < "2020-12-01")]
        zeile_aktuelles_monatesende = relevante_monatsenden[(relevante_monatsenden["Monat"] == aktuelles_datum.month) & (relevante_monatsenden["Jahr"] == aktuelles_datum.year)].index
        
        zeitreihe = data[-zeile_aktuelles_monatesende[0]-755:-zeile_aktuelles_monatesende[0]]["adjustierung"] #nachdem das aktuelle Monatsende gefunden wurde, wird die Zeitreihe enstprechend zurechtgestutzt
        zeitreihe = zeitreihe.dropna() #alle Einträge mit NAs werden gelöscht

        if len(zeitreihe) < 755: #wenn die Zeitreihe durch dropna gekürzt wurde, wird keine Vorhersage erstellt
            Aktie = mitglied
            aktueller_kurs = "NA"
            vorhersage = "NA"
            prozentuale_Änderung = "NA"
            tatsächlich = "NA"
            absolute_abweichung = "NA"
            realtive_abweichung = "NA"
            datum = aktuelles_datum            
        else:
	    #Im folgenden Codeblog wird die Zeitreihe mit dem MinMaxScaler auf Werte zwischen 0 und 1 skaliert
            daten_scaled = np.reshape(zeitreihe.values, (len(zeitreihe),1))
            scaler = MinMaxScaler(feature_range=(0, 1))
            daten_scaled = scaler.fit_transform(daten_scaled)
            liste= list(daten_scaled.flatten())
            #definieren der Variablen für die Architektur des CNN
            n_steps_out= 21 #Länge des Vektors der Prognose
            n_steps_in = 126 #Länge des Input-Vektors
            n_features = 1 #Anzahl der erklärenden Varbiablen
            X, y = split_sequence(liste, n_steps_in,n_steps_out) #Die Zeitreihe wird mit der split_squence Funktion in Input und Outputverktoren squenziert
            X = X.reshape((X.shape[0], X.shape[1], n_features))
            #In den folgenden Zeilen wird das LSTM-Modell zusammengestellt
            model = Sequential()
            model.add(LSTM(128, input_shape=(n_steps_in, n_features))) #Einfügen der LSTM-Schicht
            model.add(Dense(n_steps_out)) #Erstellen der Dense-Layer
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=10 ,batch_size= 10)
            #In den folgenden Zeilen wird die Prognose erstellt 
            x_input = np.array(liste[-126:]) #Vorbereiten des Inputs für die Prognose
            x_input = x_input.reshape((1, n_steps_in, n_features))
            yhat = model.predict(x_input, verbose=0)
	    #Verarbeitung der Daten, um sie in die Ergebnisstabelle einzufügen
            Aktie = mitglied
            aktueller_kurs = zeitreihe.values[-1]
            vorhersage = scaler.inverse_transform(yhat)[-1,-1] #macht die MinMax-Skalierung rückgängig
            prozentuale_Änderung = vorhersage/aktueller_kurs-1
            tatsächlich = float(data[-zeile_aktuelles_monatesende[0]:-zeile_aktuelles_monatesende[0]+21]['adjustierung'].tail(1))
            absolute_abweichung = tatsächlich-vorhersage
            realtive_abweichung = vorhersage/tatsächlich-1
            datum = aktuelles_datum
            
            ergebnisse = [Aktie, aktueller_kurs, vorhersage, prozentuale_Änderung, tatsächlich, absolute_abweichung, realtive_abweichung, datum] #überführen der einzelnen Variablen in eine Liste
            df = pd.Series(ergebnisse, index=zwischenergebnis_matrix.columns) #Überführen der Liste in eine pandas.Series
            zwischenergebnis_matrix = zwischenergebnis_matrix.append(df, ignore_index=True) #Einfügen der Series in die Ergebnistabelle
            
    zwischenergebnis_matrix.to_csv(os.path.join(r"D:\Statistik Masterarbeit\Daten\Ergebnisse_CNN", str(datum.date()).replace("\\","/"))) #Die Tabelle wird im gegebenen Verzeichnis abgespeichert
     
