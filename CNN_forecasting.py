# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 22:28:18 2021

@author: janhe
"""


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

# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

dax_mitglieder = pd.read_csv(r"C:\Users\janhe\Documents\Masterarbeit Statistik\Daten\DAX_Liste_final.csv", sep=";", header=None)
dax_mitglieder[0] = pd.to_datetime(dax_mitglieder[0])
dax_mitglieder.set_index(0, inplace=True)


for i in range(0,len(dax_mitglieder)): #len(dax_mitglieder)
    aktuelle_mitglieder = list(dax_mitglieder.iloc[i])
    aktuelles_datum = dax_mitglieder.index[i]

    #zwischenergebnis_matrix = np.zeros((len(aktuelle_mitglieder), 8))
    zwischenergebnis_matrix = pd.DataFrame(columns = ["Aktie", "aktueller_kurs", "vorhersage", "prozentuale_Änderung", "tatsächlich", "absolute_abweichung", "realtive_abweichung", "datum"], dtype=object)

    for mitglied in aktuelle_mitglieder:
        print(mitglied, aktuelle_mitglieder.index(mitglied), i)
        data  = pd.read_csv(os.path.join(r"D:\Statistik Masterarbeit\Daten\Aktienkurse_adjustiert", mitglied+".csv.csv").replace("\\","/"))[["adjustierung", "Datum", "RIC"]]
        data['Datum'] = pd.to_datetime(data['Datum'])
        data = data.sort_values('Datum', ascending=True)
        
        hilfs_df = data
        hilfs_df['Monat'] = hilfs_df['Datum'].dt.month
        hilfs_df['Monat_shift'] = hilfs_df['Monat'].shift(-1)
        hilfs_df['Jahr'] = hilfs_df['Datum'].dt.year
        alle_monatsenden = hilfs_df[hilfs_df['Monat'] != hilfs_df['Monat_shift']]
        relevante_monatsenden = alle_monatsenden[(hilfs_df['Datum'] > "2017-12-01") & (hilfs_df['Datum'] < "2020-12-01")]
        zeile_aktuelles_monatesende = relevante_monatsenden[(relevante_monatsenden["Monat"] == aktuelles_datum.month) & (relevante_monatsenden["Jahr"] == aktuelles_datum.year)].index
        
        zeitreihe = data[-zeile_aktuelles_monatesende[0]-755:-zeile_aktuelles_monatesende[0]]["adjustierung"]
        zeitreihe = zeitreihe.dropna()


        if zeitreihe.isna().sum() > 0:
            Aktie = mitglied
            aktueller_kurs = "NA"
            vorhersage = "NA"
            prozentuale_Änderung = "NA"
            tatsächlich = "NA"
            absolute_abweichung = "NA"
            realtive_abweichung = "NA"
            datum = aktuelles_datum
        if len(zeitreihe) < 755:
            Aktie = mitglied
            aktueller_kurs = "NA"
            vorhersage = "NA"
            prozentuale_Änderung = "NA"
            tatsächlich = "NA"
            absolute_abweichung = "NA"
            realtive_abweichung = "NA"
            datum = aktuelles_datum            
        else:
            daten_scaled = np.reshape(zeitreihe.values, (len(zeitreihe),1))
            scaler = MinMaxScaler(feature_range=(0, 1))
            daten_scaled = scaler.fit_transform(daten_scaled)
            liste= list(daten_scaled.flatten())
            
 
            n_steps_out= 21
            n_steps_in = 126
            n_features = 1

            X, y = split_sequence(liste, n_steps_in,n_steps_out)

            X = X.reshape((X.shape[0], X.shape[1], n_features))
            # define model
            model = Sequential()
            model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(50, activation='relu'))
            model.add(Dense(n_steps_out))
            model.compile(optimizer='adam', loss='mse')
        
           
            model.fit(X, y, epochs=10, batch_size= 10)
            
           
            # demonstrate prediction
            x_input = np.array(liste[-126:])
            x_input = x_input.reshape((1, n_steps_in, n_features))
            yhat = model.predict(x_input, verbose=0)
            print(yhat)
            
            Aktie = mitglied
            aktueller_kurs = zeitreihe.values[-1]
            vorhersage = scaler.inverse_transform(yhat)[-1,-1]
            prozentuale_Änderung = vorhersage/aktueller_kurs-1
            tatsächlich = float(data[-zeile_aktuelles_monatesende[0]:-zeile_aktuelles_monatesende[0]+21]['adjustierung'].tail(1))
            absolute_abweichung = tatsächlich-vorhersage
            realtive_abweichung = vorhersage/tatsächlich-1
            datum = aktuelles_datum
            
            ergebnisse = [Aktie, aktueller_kurs, vorhersage, prozentuale_Änderung, tatsächlich, absolute_abweichung, realtive_abweichung, datum]
            df = pd.Series(ergebnisse, index=zwischenergebnis_matrix.columns)
            zwischenergebnis_matrix = zwischenergebnis_matrix.append(df, ignore_index=True)
            
    zwischenergebnis_matrix.to_csv(os.path.join(r"D:\Statistik Masterarbeit\Daten\Ergebnisse_CNN", str(datum.date()).replace("\\","/")))
     
    
    
    
    
    
    
    
    
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features), dilation_rate =2))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
        
        
        
        
        
