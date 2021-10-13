# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 14:15:39 2021

@author: janhe
"""


import numpy as np
import pandas as pd
from numpy import array
import os

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Activation, Dropout
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


i=3
mitglied = aktuelle_mitglieder[1]

for i in range(31,len(dax_mitglieder)): #len(dax_mitglieder)
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
            #model.add(LSTM(126, activation='relu', return_sequences=False, input_shape=(n_steps_in, n_features)))
            model.add(LSTM(128, input_shape=(n_steps_in, n_features)))
            model.add(Dense(n_steps_out))
            model.compile(optimizer='adam', loss='mse')
           
            model.fit(X, y, epochs=10 ,batch_size= 10)
            
           
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
            
    zwischenergebnis_matrix.to_csv(os.path.join(r"D:\Statistik Masterarbeit\Daten\Ergebnisse_LSTM", str(datum.date()).replace("\\","/")))
     
    
        
        
        
        
        
        
        
d = pd.read_csv(os.path.join(r"D:\Statistik Masterarbeit\Daten\Ergebnisse_LSTM", str(datum.date()).replace("\\","/")))
            
            zwischenergebnis_matrix2 = zwischenergebnis_matrix
            
    str(datum.date())
            len(ergebnisse)
            zwischenergebnis_matrix[0] = ergebnisse
            scaler.inverse_transform(yhat)

zwischenergebnis_matrix = zwischenergebnis_matrix.append(df, ignore_index=True)
df = pd.Series(ergebnisse, index=zwischenergebnis_matrix.columns)
.loc[1] = ergebnisse 

zeitreihe.plot()
plt.plot(liste)

zwischenergebnis_matrix = zwischenergebnis_matrix.append(ergebnisse)
float(data[-zeile_aktuelles_monatesende[0]:-zeile_aktuelles_monatesende[0]+21]['adjustierung'].tail(1))
            scaler.inverse_transform(yhat)[-1,-1]


X = pd.dropna(zeitreihe)

np.where(pd.isna(newlist))

liste = [x for x in liste if np.isnan(x) == False]

pd.isna(newlist)

liste[415]
.sum()

pd.isna(zeitreihe)
zeitreihe.dropna()



liste.dropna()

from keras import optimizers

optimizer = optimizers.Adam(clipvalue=0.5, clipnorm)


            scaler.inverse_transform(liste)


import matplotlib.pyplot as plt
plt.plot(data[-zeile_aktuelles_monatesende[0]-755:-zeile_aktuelles_monatesende[0]]["adjustierung"])
plt.plot(daten_scaled)            
            
        
 
           #Zeilen mit Monatsende finden
    hilfs_df <- stock #überführen der Daten in einen Hilfsdatensatz
    hilfs_df['Date'] = as.Date(hilfs_df$Datum, "%Y-%m-%d") #Spalte mit Datum einfügen
    #row.names(hilfs_df) <- NULL #Index wird neu gesetzt
    hilfs_df['Month'] <- month(as.Date(format(hilfs_df$Date, "%Y-%m-%d"))) #Einfügen einer Spalte mit Monat
    hilfs_df['Month_shift']<- shift(month(as.Date(format(hilfs_df$Date, "%Y-%m-%d"))),-1) #Einfügen einer Spalte mit Monat um eins nach oben verschoben
    alle_monatsende <- subset(hilfs_df, hilfs_df['Month'] != hilfs_df['Month_shift']) #es werden die Zeilen extrahiert, wo Monat ungleich Monag um eins verschoben
    monatsende <- subset(alle_monatsende, as.Date(format(alle_monatsende$Date, "%Y-%m-%d")) > "2017-12-01" & as.Date(format(alle_monatsende$Date, "%Y-%m-%d")) < "2020-12-01") #es werden nur die für die Studie relevanten Monate extrahiert
    zeile_aktuelles_monatsende <- as.integer(row.names(monatsende[which(year(monatsende$Datum) == year(rownames(mitglieder_liste)[t]) & month(monatsende$Datum) == month(rownames(mitglieder_liste)[t])),])) #findet den letzten Handelstag des Monats ausgehend von der aktuellen DAX-Liste
    
    zeitreihe <- stock[(zeile_aktuelles_monatsende[1]-756):zeile_aktuelles_monatsende[1],]['adjustierung'] #extrahieren der letzten 252 Handelstage
    colnames(zeitreihe)[1] <- "Kurs" #Spalte umbenennen
    
    

data['Datum'] = pd.to_datetime(data['Datum'])

daten= data['adjustierung'].tail(750)
daten_scaled = np.reshape(daten.values, (len(daten),1))
scaler = MinMaxScaler(feature_range=(0, 1))
daten_scaled = scaler.fit_transform(daten_scaled)
liste= list(daten_scaled.flatten())

i=0


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




n_steps_out=21
n_steps_in = 126
n_features = 1


X, y = split_sequence(liste, n_steps_in,n_steps_out)
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(126, activation='relu', return_sequences=False, input_shape=(n_steps_in, n_features)))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')

start.time
time
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
# fit model
model = model.fit(X, y, epochs=10,batch_size= 10 , verbose=0)

import timeit

start = timeit.default_timer()
model = model.fit(X, y, epochs=10,batch_size= 10 , verbose=0)

stop = timeit.default_timer()

print('Time: ', stop - start)  

model.history



from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
 

# define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features), dilation_rate =2))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')

import timeit

start = timeit.default_timer()
model.fit(X, y, epochs=10, verbose=0)

stop = timeit.default_timer()

print('Time: ', stop - start)  









# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# split into samples
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=50, verbose=0)

    # reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))



# define model
model = keras.Sequential()
model.add(LSTM(10, activation='relu', return_sequences=False, input_shape=(n_steps_in, n_features)))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')


# fit model
t = model.fit(X, y, epochs=10, verbose=0)




liste = 

daten= data['adjustierung'].tail(750).values
daten = daten.reshape((len(daten), 1))

sc = MinMaxScaler(feature_range = (0, 1))
liste_scaled = sc.transform(daten)
liste = liste_scaled

X = []
y = []
for i in range(80, 700):
    X_train.append(liste[i-80:i, 0])
    y_train.append(liste[i, 0])
X, y = np.array(X_train), np.array(y_train)


X, y = split_sequence(liste, n_steps_in, n_steps_out)

    
    

# demonstrate prediction
x_input = np.array(data['adjustierung'].tail(126))
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

t.history
