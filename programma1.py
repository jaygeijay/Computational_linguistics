
import sys
import re
import nltk

def ATesto(frasi):
    #lunghezza totale del testo
    lunghezzaTotale=0.0
    #lista dei token 
    listaToken = []
    #lista delle POS
    tokenPOSTot = []
    for frase in frasi:
        #divido in token la frase
        token = nltk.word_tokenize(frase)
        #lista con tutti i token del testo
        tokenPOS = nltk.pos_tag(token)
        #Calcolo tutti i token del testo
        listaToken = listaToken + token
        #calcolo tutti i POS del testo
        tokenPOSTot = tokenPOSTot + tokenPOS
        lunghezzaTotale = lunghezzaTotale + len(token) 
    #restituisco la lunghezza totale del testo e la lunghezza dei token 
    return lunghezzaTotale, listaToken, tokenPOSTot


def lenavgsentences (frasi, token):
    #calcolo la media tra il numero dei token e delle frasi 
    lenavgsentences = float(token)/float(frasi)
    return lenavgsentences

def lenavgwords (token,listaToken): 
    #variabile che conta i caratteri
    nchar = 0
    #per ogni token calcolo il numero dei caratteri
    for tok in listaToken:
        nchar = nchar + len(tok)
     #calcolo la media tra il numero dei token e dei caratteri 
    lenavgwords = float(nchar)/float(token)
    return lenavgwords

def contahapax (listaToken): #funzione per calcolare gli hapax sui primi 1000 token
    h=0
    #per ogni token presente nel vocabolario
    for tok in listaToken:
        #calcolo la frequenza del token
        freqToken = listaToken.count(tok)
        #se la frequenza del token e' 1 aggiungo il token agli hapax 
        if freqToken == 1:
            h += 1
    #restituisco il numero degli hapax
    return h
    
def hapaxvoc (listaToken): #funzione come quella sopra per contare gli hapax 
    hap=0
    #calcolo il vocabolario
    vocabolario = set(listaToken)
    #per ogni token presente nel vocabolario
    for tok in vocabolario:
        #calcolo la frequenza del token
        freqToken = listaToken.count(tok)
        #se la frequenza del token è 1 aggiungo il token agli hapax 
        if freqToken == 1:
            hap += 1
    #restituisco il numero degli hapax
    return hap
    
def vocaricc(lunghezzaTotale, listaToken):
    #variabile che incrementa di 500 unità
    n = 500
    #variabili che dividono il corpus in porzioni di 500 token
    voc500 = []
    voc1000 = []
    voc1500 = []
    voc2000 = []
    voc2500 = []
    voc3000 = []
    voc3500 = []
    voc4000 = []
    voc4500 = []
    voc5000 = []
    while n < lunghezzaTotale: 
        #calcolo la grandezza del vocabolario
        vocabolario = set(listaToken[:n])
        #per ogni porzione di 1000 token calcolo la grandezza del vocabolario e il numero degli hapax (richiamando la funzione CalcolaHapax)
        if n == 500:
            voc500 = len(vocabolario)
            hap500 = hapaxvoc(listaToken[:n])
        if n == 1000:
            voc1000 = len(vocabolario)
            hap1000 = hapaxvoc(listaToken[:n])
        if n == 1500:
            voc1500 = len(vocabolario)
            hap1500 = hapaxvoc(listaToken[:n])
        if n == 2000:
            voc2000 = len(vocabolario)
            hap2000 = hapaxvoc(listaToken[:n])
        if n == 2500:
            voc2500 = len(vocabolario)
            hap2500 = hapaxvoc(listaToken[:n])
        if n == 3000:
            voc3000 = len(vocabolario)
            hap3000 = hapaxvoc(listaToken[:n])
        if n == 3500:
            voc3500 = len(vocabolario)
            hap3500 = hapaxvoc(listaToken[:n])
        if n == 4000:
            voc4000 = len(vocabolario)
            hap4000 = hapaxvoc(listaToken[:n])
        if n == 4500:
            voc4500 = len(vocabolario)
            hap4500 = hapaxvoc(listaToken[:n])    
        if n == 5000:
            voc5000 = len(vocabolario)
            hap5000 = hapaxvoc(listaToken[:n])
        #incremento la variabile e restituisco i risultati
        n += 500
    return voc500, hap500, voc1000, hap1000, voc1500, hap1500, voc2000, hap2000, voc2500, hap2500, voc3000, hap3000, voc3500, hap3500, voc4000, hap4000, voc4500, hap4500, voc5000, hap5000

#funzione per il calcolo della ttr
def TTR(hapax, voc):
    #calcolo la Type Token Ratio
    ttr = float(hapax)/float(voc)
    return ttr
    
    
#funzione che calcola la distribuzione in percentuale dei sostantivi, degli aggettivi, dei verbi e dei pronomi per frase
def percentuale(tokenPOSTot, frasi):
    #variabili che contano il numero dei sostantivi, degli aggettivi, dei verbi e dei pronomi
    Numero_Sostantivi = 0
    Numero_Aggettivi = 0
    Numero_Verbi = 0
    Numero_Avverbi = 0
    #per ogni token contenuto all'interno delle POS calcolo il numero delle varie POS e incremento le variabili
    for token in tokenPOSTot:
        #sostantivi
        if token[1] in {"NN", "NNS", "NNP", "NNPS"}:
            Numero_Sostantivi += 1
        #aggettivi
        if token[1] in {"JJ", "JJR", "JJS"}:
            Numero_Aggettivi += 1
        #verbi
        if token[1] in {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}:
            Numero_Verbi +=  1
        #avverbi
        if token[1] in {"RB", "RBS", "RBR", "WRB"}: 
           Numero_Avverbi += 1
    
    #Calcolo la percetuale dei Sostantivi, degli aggettivi, dei verbi e dei pronomi
    Percentuale_Sostantivi = float(Numero_Sostantivi/100)
    Percentuale_Aggettivi = float(Numero_Aggettivi/100)
    Percentuale_Verbi = float(Numero_Verbi/100)
    Percentuale_Avverbi = float(Numero_Avverbi/100)
    

    return Percentuale_Sostantivi, Percentuale_Aggettivi, Percentuale_Verbi, Percentuale_Avverbi  
    
    #funzione che calcola la distribuzione in percentuale parole funzionali
def percentage(tokenPOSTot, frasi):
    #variabili che contano il numero 
    Numero_Articoli = 0
    Numero_Preposizioni = 0
    Numero_Congiunzioni = 0
    Numero_Pronomi = 0
    #per ogni token contenuto all'interno  calcolo il numero e incremento le variabili
    for token in tokenPOSTot:
        #sostantivi
        if token[1] in {"WDT", "DT", "PDT"}:
            Numero_Articoli += 1
        #aggettivi
        if token[1] in {"IN"}:
            Numero_Preposizioni += 1
        #verbi
        if token[1] in {"CC"}:
            Numero_Congiunzioni +=  1
        #avverbi
        if token[1] in {"PRP", "PRP$", "RP"}: 
           Numero_Pronomi += 1
    
    #Calcolo la percetuale dei Sostantivi, degli aggettivi, dei verbi e dei pronomi
    Percentuale_Articoli = float(Numero_Articoli/100)
    Percentuale_Preposizioni = float(Numero_Preposizioni/100)
    Percentuale_Congiunzioni = float(Numero_Congiunzioni/100)
    Percentuale_Pronomi = float(Numero_Pronomi/100)
    

    return Percentuale_Articoli, Percentuale_Preposizioni, Percentuale_Congiunzioni, Percentuale_Pronomi  

def main(file1, file2):
    #apro i due file
    fileInput1 = open(file1, mode="r", encoding="utf-8")
    fileInput2 = open(file2, mode="r", encoding="utf-8")   
    #leggo i due file
    raw1 = fileInput1.read()
    raw2 = fileInput2.read()
    #divido in token 
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #divido in frasi i due file
    frasi1 = sent_tokenizer.tokenize(raw1)
    frasi2 = sent_tokenizer.tokenize(raw2)
    #confronto le lunghezze dei 2 file
    ntoken1, listaToken1, tokenPOSTot1 = ATesto(frasi1)
    ntoken2, listaToken2, tokenPOSTot2 = ATesto(frasi2)    
    #numero delle frasi
    nfrasi1 = len(frasi1)
    nfrasi2 = len(frasi2)
    #lunghezza media delle frasi in termini di token
    lenavgsentences1 = lenavgsentences(nfrasi1, ntoken1)
    lenavgsentences2 = lenavgsentences(nfrasi2, ntoken2)
    #lunghezza media delle parole in termini di caratteri senza punteggiatura
    tokenizer=nltk.RegexpTokenizer(r"\w+") #Regexptokenizer restituisce testo come una lista di parole con la punteggiatura rimossa
    listaToken1nopunt=tokenizer.tokenize(raw1) #assegno a listatokennopunt i valori restituiti senza punteggiatura
    listaToken2nopunt=tokenizer.tokenize(raw2)
    ntoken1nopunt, lista1nopunt, tokenPOSTot1 = ATesto(listaToken1nopunt)
    ntoken2nopunt, lista2nopunt, tokenPOSTot2 = ATesto(listaToken2nopunt) 
    lenavgwords1 = lenavgwords(ntoken1nopunt, listaToken1nopunt)
    lenavgwords2= lenavgwords(ntoken2nopunt, listaToken2nopunt)
    #calcolo hapax sui primi 1000 token
    
    hapax1=contahapax(listaToken1[:1000]) 
    hapax2=contahapax(listaToken2[:1000])
    
    #grandezza del vocabolario e richezza lessicale calcolata attraverso la TTR, calcolati all'aumentare del corpus per porzioni da 500token
    voc500_1, hap500_1, voc1000_1, hap1000_1, voc1500_1, hap1500_1, voc2000_1, hap2000_1, voc2500_1, hap2500_1, voc3000_1, hap3000_1, voc3500_1, hap3500_1, voc4000_1, hap4000_1, voc4500_1, hap4500_1, voc5000_1, hap5000_1 = vocaricc(ntoken1, listaToken1)
    voc500_2, hap500_2, voc1000_2, hap1000_2, voc1500_2, hap1500_2, voc2000_2, hap2000_2, voc2500_2, hap2500_2, voc3000_2, hap3000_2, voc3500_2, hap3500_2, voc4000_2, hap4000_2, voc4500_2, hap4500_2, voc5000_2, hap5000_2 = vocaricc(ntoken2, listaToken2)
    
    #percentuale parole funzionali
    PercentualeArticoli_1, PercentualePreposizioni_1, PercentualeCongiunzioni_1, PercentualiPronomi_1 = percentage(tokenPOSTot1, nfrasi1)
    PercentualeArticoli_2,  PercentualePreposizioni_2,  PercentualeCongiunzioni_2,  PercentualiPronomi_2 = percentage(tokenPOSTot2, nfrasi2)   
    #percentuale parole piene
    PercentualeSostantivi_1, PercentualeAggettivi_1, PercentualeVerbi_1, PercentualiAvverbi_1 = percentuale(tokenPOSTot1, nfrasi1)
    PercentualeSostantivi_2,  PercentualeAggettivi_2,  PercentualeVerbi_2,  PercentualiAvverbi_2 = percentuale(tokenPOSTot2, nfrasi2)   
    
    #numero delle frasi
    print ("Confronto i due corpora per numero di frasi:\n")
    print (file1, "contiene:\t", nfrasi1, "frasi")
    print (file2, "contiene:\t", nfrasi2, "frasi")
    if nfrasi1 > nfrasi2:
        print (file1, "ha più frasi del corpus", file2, "\n")
    elif nfrasi2 > nfrasi1:
        print (file2, "ha più frasi del corpus", file1, "\n")
    else:
        print ("I due corpora hanno lo stesso numero di frasi\n")
    print ("\n")    
        
    #numero dei token
    print ("Confronto i due corpora per numero di token:\n")
    print (file1, "contiene:\t", ntoken1, "token")
    print (file2, "contiene:\t", ntoken2, "token")
    if ntoken1 > ntoken2:
        print (file1, "ha più token del corpus", file2, "\n")
    elif ntoken2 > ntoken1:
        print (file2, "ha più token del corpus", file1, "\n")
    else:
        print ("I due corpora hanno lo stesso numero di token\n")
    print ("\n")
    #lunghezza media frasi in token
    print ("lunghezza media delle frasi in termini di token\n")
    print ("Lunghezza media delle frasi in termini di token di", file1, ":", lenavgsentences1)
    print ("Lunghezza media delle frasi in termini di token di", file2, ":", lenavgsentences2)
    print ("\n")
    print ("\n")
    #lunghezza media in caratteri escludendo la punteggiatura
    print ("lunghezza media dei token in termini di caratteri escludendo la punteggiatura\n")
    print ("Lunghezza media dei token in termini di caratteri senza punteggiatura di", file1, ":", lenavgwords1)
    print ("Lunghezza media dei token in termini di caratteri senza punteggiatura di", file2, ":", lenavgwords2)
    print ("\n")
    print ("\n")
    #calcolo degli hapax
    print("numero di hapax sui primi 1000 token\n")
    print("il valore degli hapax in", file1, "è:", hapax1)
    print("il valore degli hapax in", file2, "è:", hapax2)
    print ("\n")
    print ("\n")
    #grandezza del vocabolario all'aumentare del corpus per porzioni da 500token
    print ("Vocabolario all'aumentare del corpus di", file1, "\n") 
    print ("\t", voc500_1)
    print ("\t", voc1000_1)
    print ("\t", voc1500_1)
    print ("\t", voc2000_1)
    print ("\t", voc2500_1)
    print ("\t", voc3000_1)
    print ("\t", voc3500_1)
    print ("\t", voc4000_1)
    print ("\t", voc4500_1)
    print ("\t", voc5000_1)
    print ("\n")
    
    
    print ("Vocabolario all'aumentare del corpus di", file2, "\n") 
    print ("\t", voc500_2)
    print ("\t", voc1000_2)
    print ("\t", voc1500_2)
    print ("\t", voc2000_2)
    print ("\t", voc2500_2)
    print ("\t", voc3000_2)
    print ("\t", voc3500_2)
    print ("\t", voc4000_2)
    print ("\t", voc4500_2)
    print ("\t", voc5000_2)
    print ("\n")
    
   
    #ricchezza lessicale calcolata attraverso la ttr porzioni da 500 token
    print ("ricchezza lessicale calcolata attraverso la ttr in porzioni da 500 token", file1)
    print ("\t ricchezza lessicale 500 token:", TTR(hap500_1, voc500_1))
    print ("\t ricchezza lessicale 1000 token:", TTR(hap1000_1, voc1000_1))
    print ("\t ricchezza lessicale 1500 token:", TTR(hap1500_1, voc1500_1))
    print ("\t ricchezza lessicale 2000 token:", TTR(hap2000_1, voc2000_1))
    print ("\t ricchezza lessicale 2500 token:", TTR(hap2500_1, voc2500_1))
    print ("\t ricchezza lessicale 3000 token:", TTR(hap3000_1, voc3000_1))
    print ("\t ricchezza lessicale 3500 token:", TTR(hap3500_1, voc3500_1))
    print ("\t ricchezza lessicale 4000 token:", TTR(hap4000_1, voc4000_1))
    print ("\t ricchezza lessicale 4500 token:", TTR(hap4500_1, voc4500_1))
    print ("\t ricchezza lessicale 5000 token:", TTR(hap5000_1, voc5000_1))
    print ("\n")
    
    print ("ricchezza lessicale calcolata attraverso la ttr in porzioni da 500 token", file2)
    print ("\t ricchezza lessicale 500 token:", TTR(hap500_2, voc500_2))
    print ("\t ricchezza lessicale 1000 token:", TTR(hap1000_2, voc1000_2))
    print ("\t ricchezza lessicale 1500 token:", TTR(hap1500_2, voc1500_2))
    print ("\t ricchezza lessicale 2000 token:", TTR(hap2000_2, voc2000_2))
    print ("\t ricchezza lessicale 2500 token:", TTR(hap2500_2, voc2500_2))
    print ("\t ricchezza lessicale 3000 token:", TTR(hap3000_2, voc3000_2))
    print ("\t ricchezza lessicale 3500 token:", TTR(hap3500_2, voc3500_2))
    print ("\t ricchezza lessicale 4000 token:", TTR(hap4000_2, voc4000_2))
    print ("\t ricchezza lessicale 4500 token:", TTR(hap4500_2, voc4500_2))
    print ("\t ricchezza lessicale 5000 token:", TTR(hap5000_2, voc5000_2))
    print ("\n")
    print ("\n")
    
    
    print ("Percentuale dell'insieme delle parole piene (aggettivi, sostantivi, verbi e avverbi) in", file1)
    print ("\taggettivi:", PercentualeAggettivi_1,"%")
    print ("\tsostantivi:", PercentualeSostantivi_1,"%")
    print ("\tverbi:", PercentualeVerbi_1,"%")
    print ("\tavverbi:", PercentualiAvverbi_1,"%")
    print  ("\n")
    
    print ("Percentuale dell'insieme delle parole piene (aggettivi, sostantivi, verbi e avverbi) in", file2)
    print ("\taggettivi:", PercentualeAggettivi_2,"%")
    print ("\tsostantivi:", PercentualeSostantivi_2,"%")
    print ("\tverbi:", PercentualeVerbi_2,"%")
    print ("\tavverbi:", PercentualiAvverbi_2,"%")
    print ("\n")
    
    print ("Percentuale dell'insieme delle parole funzionali (articoli, preposizioni, congiunzioni e pronomi) in", file1)
    print ("\tarticoli:", PercentualeArticoli_1,"%")
    print ("\tpreposizioni:", PercentualePreposizioni_1,"%")
    print ("\tcongiunzioni:", PercentualeCongiunzioni_1,"%")
    print ("\tpronomi:", PercentualiPronomi_1,"%")
    print  ("\n")
    
    print ("Percentuale dell'insieme delle parole funzionali (articoli, preposizioni, congiunzioni e pronomi) in", file2)
    print ("\tarticoli:", PercentualeArticoli_2,"%")
    print ("\tpreposizioni:", PercentualePreposizioni_2,"%")
    print ("\tcongiunzioni:", PercentualeCongiunzioni_2,"%")
    print ("\tpronomi:", PercentualiPronomi_2,"%")
    print ("\n")
    
    
    
    
    
main(sys.argv[1], sys.argv[2])
