library(rugarch)
library(tseries)
library(ggplot2)
library(lubridate)
library(data.table)

mitglieder_liste <- read.csv("C:/Users/anaconda/Desktop/DAX_Liste_final.csv", sep=";", header=FALSE, stringsAsFactors=FALSE)

mitglieder_liste[1,1] <- "30.12.2017" 
rownames(mitglieder_liste) <- as.Date(mitglieder_liste$V1, "%d.%m.%Y")
mitglieder_liste$V1 <- NULL

ergebnisse <- array(data = 0, dim = c(30,9,36)) 
colnames(ergebnisse) <- c("Aktie", "Datum", "aktueller Kurs", "Prognose", "tatsächlich","prozentuale Änderung", "absolute Abweichung", "relative Abweichung", "Methode")



for (t in 1:length(rownames(mitglieder_liste))){
  print(rownames(mitglieder_liste)[t])
  aktuelle_mitglieder <- as.list(mitglieder_liste[t,])
  zwischenergebnis_matrix = array(data = 0, dim = c(length(aktuelle_mitglieder),9))
  rownames(zwischenergebnis_matrix) <- aktuelle_mitglieder
  colnames(zwischenergebnis_matrix) <- c("Aktie", "Datum","aktueller Kurs", "Prognose","tatsächlich", "prozentuale Änderung", "absolute Abweichung", "relative Abweichung", "Methode")
  for (mitglied in aktuelle_mitglieder){
    print(mitglied)
    stock <- read.csv(file = file.path("C:/Users/anaconda/Desktop/Aktienkurse_adjustiert", paste0(paste(" ", mitglied),".csv", ".csv")))[c('Datum', 'adjustierung', 'RIC')] #einlesen der Daten
    stock <- stock[order(stock$Datum),] #sortieren der Daten
    row.names(stock) <- NULL
    
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

    
    if (sum(is.na(zeitreihe)>0)){
      Aktie = as.character(stock$RIC[1])
      aktueller_kurs <- "NA"
      vorhersage = "NA"
      prozentuale_Änderung = "NA"
      tatsächlich = "NA"
      absolute_abweichung = "NA"
      realtive_abweichung = "NA"
      Methode = "NA"
      datum = as.character(stock[zeile_aktuelles_monatsende,]$Datum)
    }else{
      
      #Daten stationarisieren
      zeitreihe_stationär <- ts(zeitreihe) #Zeitreihe in Format ts umwandeln
      d = 0 #Zähler für Ordnung d setzen
      
      while (adf.test(zeitreihe_stationär)$p.value > 0.05){ #d-maliges Stationarisieren bis ADF-Test bei alpha=0.05 Stationarität signalisiert
        zeitreihe_stationär<- diff(zeitreihe_stationär)
        d = d+1
      }
      
      
      #Bestes ARMA-Modell finden
      final.aic <- Inf
      final.order <- c(0,0,0)
      for (p in 0:5) for (q in 0:5) {
        arimaFit = tryCatch( arima(zeitreihe_stationär, order=c(p, 0, q)),
                             error=function( err ) FALSE,
                             warning=function( err ) FALSE )
        
        if( !is.logical( arimaFit ) ) {
          current.aic <- AIC(arimaFit)
          if (current.aic < final.aic) {
            final.aic <- current.aic
            final.order <- c(p, 0, q)
            final.arima <- arima(zeitreihe_stationär, order=final.order)
          }
        } else {
          next
        }
      }
      
      
      #Test auf GARCH-Effekt
      
      Box.test(resid(final.arima),lag=21,type="Ljung-Box", fitdf = p+q)$p.value #Ljung-Box Test der Residuen für das ARMA-Modell mit p+q Freiheitsgraden und Anzahl der Lags gleich des Vorhersagehorizonts
      
      #https://www.sciencedirect.com/science/article/abs/pii/S0378437119320618
      
      #Falls Vorhanden: GARCH fitten
      if (Box.test(resid(final.arima),lag=21,type="Ljung-Box", fitdf = p+q)$p.value<0.05){
        final.aic <- Inf
        final.order <- c(0,0,0,0,0)
        for (p in 0:5) for (q in 0:5) for (P in 1:2) for (Q in 1:2) {
          tryCatch(
            {
              model=ugarchspec(
                variance.model = list(model = "sGARCH", garchOrder = c(P, Q)),
                mean.model = list(armaOrder = c(p, q), include.mean = F),
                distribution.model = "std")
              
              modelfit=ugarchfit(spec=model,data=zeitreihe_stationär)
              
              current.aic <- infocriteria(modelfit)[1]
              
            },
            warning=function(cond) {
              message("Modell konvergiert nicht")
              message("Hier ist die originale Fehlermeldung")
              message(cond)
              current.aic <- 999
            }
          )
          
          if (current.aic < final.aic) {
            final.aic <- current.aic
            final.order <- c(p, 0, q, P, Q)
          }
          else {
            next
          } }
        forecast = ugarchforecast(modelfit,n.ahead=21)@forecast$seriesFor
        Methode = "ARIMA-GARCH"
      } else {
        forecast = predict(final.arima, n.ahead=21)$pred
        Methode = "ARIMA"
      }
      
      Aktie = as.character(stock$RIC[1])
      if (d>0){
        vorhersage = tail(tail(zeitreihe,1)$Kurs+tail(diffinv(forecast,differences = d),21),1)
      }else{
      vorhersage = tail(forecast, 1)
      }
      prozentuale_Änderung = vorhersage/tail(zeitreihe,1)$Kurs-1
      aktueller_kurs <- tail(zeitreihe,1)$Kurs
      tatsächlich = stock[zeile_aktuelles_monatsende+21,]$adjustierung
      absolute_abweichung = vorhersage-tatsächlich
      relative_abweichung = vorhersage/tatsächlich-1
      methode = Methode
      datum = as.character(stock[zeile_aktuelles_monatsende,]$Datum)
    }
    print(final.order)
    vector <- c(Aktie, datum, aktueller_kurs, vorhersage, tatsächlich, prozentuale_Änderung, absolute_abweichung, relative_abweichung, methode)
    zwischenergebnis_matrix[Aktie,] <- vector
  }
  ergebnisse[,,t] <- zwischenergebnis_matrix
  write.csv(zwischenergebnis_matrix, file=rownames(mitglieder_liste)[t])
}

