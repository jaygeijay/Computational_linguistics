import sys
import nltk
import re
import math
from nltk import FreqDist
from nltk import bigrams
from nltk import trigrams


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



#estrarre e ordinare in ordine di frequenza decrescente, indicando anche la relativa frequenza:
#le 10 PoS più frequenti
#i 10 bigrammi di PoS più frequenti
#i 10 trigrammi di PoS più frequenti
#i 20 aggettivi e i 20 avverbi più frequenti


#funzione che calcola i 10 POS piu' frequenti
def PoSfreq(tokenPOSTot):
    #lista che contiene le POS
    PartOfSpeech = []
    #scorro le POS le aggiungo alla lista definita precedentemente
    for token in tokenPOSTot:
        #calcolo la distribuzione di frequenza delle POS e le 10 POS piu' frequenti 
        PartOfSpeech.append(token[1])
    #calcolo la distribuzione di frequenza delle POS e le 10 POS piu' frequenti
    DistribuzionePOS = nltk.FreqDist(PartOfSpeech)
    POSFrequenti = DistribuzionePOS.most_common(10)
    return POSFrequenti


#funzione che stampa le distribuzioni di frequenza 
def frequenze(Distribuzione):
    for elem in Distribuzione:
        print ("\t", elem[0], "compare", elem[1], "volte")

def stampafreq(Dist):
    for elem in Dist:
        print ("\t", elem[0], "compare", elem[1], "volte")

#funzione che calcola la frequenza de trigrammi
def Calcolabigrammi(bigrammi):
    #calcolo la frequenza dei trigrammi di POS
    Distribuzionebigrammi = nltk.FreqDist(bigrammi)
    #estraggo i 10 trigrammi di POS più frequenti
    bigrammiFrequenti = Distribuzionebigrammi.most_common(10)
    return bigrammiFrequenti
#funzione che stampa le frequenze dei bigrammi
def frequenzabigrammi(bigrammifreq):
    for elem in bigrammifreq:
        print ("\t", elem[0][0][1], elem[0][1][1],  "compare", elem[1], "volte")


#funzione che calcola la frequenza de trigrammi
def Calcolatrigrammi(trigrammi):
    #calcolo la frequenza dei trigrammi di POS
    DistribuzioneTrigrammi = nltk.FreqDist(trigrammi)
    #estraggo i 10 trigrammi di POS più frequenti
    trigrammiFrequenti = DistribuzioneTrigrammi.most_common(10)
    return trigrammiFrequenti

#funzione che stampa le frequenze dei trigrammi
def frequenzatrigrammi(trigrammifreq):
    for elem in trigrammifreq:
        print ("\t", elem[0][0][1], elem[0][1][1], elem[0][2][1], "compare", elem[1], "volte")

#funzione che calcola i 20 aggettivi piu' frequenti
def aggettivi(tokenPOSTot):
    #lista dgli aggettivi frequenti
    AggFreq = []
    #Scorro le POS token per token
    for token in tokenPOSTot:
        #se il token è un aggettivo lo aggiungo alla lista degli aggettivi frquenti
        if token[1] in {"JJ", "JJR", "JJS"}:
            AggFreq.append(token[0])
    #calcolo la distribuzione di frequenza degli aggettivi e i 20 aggettivi più frequenti
    DistribuzioneAggettivi = nltk.FreqDist(AggFreq)
    aggettivifrequenti = DistribuzioneAggettivi.most_common(20)
    return aggettivifrequenti


#funzione che calcola i 20 avverbi piu' frequenti
def avverbi(tokenPOSTot):
    #lista degli avverbi frequenti
    aFreq = []
    #Scorro le POS token per token
    for token in tokenPOSTot:
        #se il token è un avverbio lo aggiungo alla lista degli avverbi frequenti
        if token[1] in {"RB", "RBS", "RBR", "WRB"}:
            aFreq.append(token[0])
    #calcolo la distribuzione d frequenza degli avverbi e i 20 avverbi più frequenti
    Distribuzioneavverbi = nltk.FreqDist(aFreq)
    avverbifrequenti = Distribuzioneavverbi.most_common(20)
    return avverbifrequenti




    
"""estraete ed ordinate i 20 bigrammi di token composti da aggettivo e sostantivo (dove ogni token deve avere una frequenza
maggiore di 3):
◦ con frequenza massima, indicando anche la relativa frequenza;
◦ con probabilità condizionata massima, indicando anche la relativa probabilità;
◦ con forza associativa (calcolata in termini di Local Mutual Information) massima,
indicando anche la relativa forza associativa;"""


def listaSostagg(tokenPOSTot,listaToken):
    lista=[] #lista che contiene i miei bigrammi, quindi la sequenza: aggettivo, sostantivo
    listaLMI=[] #lista della LMI cosituita da: (bigramma e LMI)
    ListaPCond= [] #lista Probabilità condizionata che sarà costituita da: bigramma, prob. condizionata
    bigrammiTokens= list(bigrams(listaToken)) 
    listab= list(bigrams(tokenPOSTot))  
    for token in listab: 
        if (token[0][1] in ["JJ", "JJR", "JJS"]) and (token[1][1] in ["NNP", "NNPS", "NN", "NNS"]): 
            frequenza1= listaToken.count(token[0][0]) 
            frequenza2= listaToken.count(token[1][0]) 
            if frequenza1 > 3 and frequenza2 > 3: 
                lista.append((token[0][0], token[1][0])) 
    bigrammii= nltk.FreqDist(lista) #utilizzo la distribuzione di frequenza considerando lista
    bigramsoggagg= bigrammii.most_common(20) #voglio i 20 più frequenti
    bigrammiDiversi= list(set(lista)) #tolgo le ripetizioni
    for token in bigrammiDiversi:
        frequBigram= bigrammiTokens.count(token) #frequenza bigrammi (che calcolo andando a contare)
        frequenzaElemento1= listaToken.count(token[0]) #frequenza primo elemento bigram
        frequenzaElemento2= listaToken.count(token[1]) #frequenza secondo elemento bigram
        ProbCondizionata= frequBigram/frequenzaElemento1 #la probabilità condizionata è P(A,B)/P(B)
        ListaPCond.append((token, ProbCondizionata)) #la aggiungo all'interno della lista 
        probabilitàtoken1= frequenzaElemento1/len(listaToken) #frequenza dell'elemento e corpus
        probabilitàtoken2= frequenzaElemento2/len(listaToken) #frequenza dell'altro elemento e corpus
        ProbCongiunta= ProbCondizionata*probabilitàtoken1 #faccio riferimento alla regola del prodotto P(A,B)=P(A)*P(B)
        Probabilità= ProbCongiunta/(probabilitàtoken1*probabilitàtoken2) #la calcolo per calcolare la MI
        MI= math.log(Probabilità,2) #gli dico che il logaritmo è in base due e che prenda in considerazione la variabile probabilità
        LMI= frequBigram*MI #sappiamo essere la MI*frequenzaosservata, quindi p(<u,v>)*MI
        listaLMI.append((token, LMI)) #la aggiungo all'interno della lista 
    Ordinamento= sorted(ListaPCond, reverse= True, key= lambda x:x[1]) #ordinamento decrescente sulla base della p. condizionata
    Pcond= Ordinamento[:20] #la soglia che non deve superare è 20 e ottengo i 20 bigrammi (bigramma, P.Cond)
    OrdinamentoLMI= sorted(listaLMI, reverse=True, key= lambda x:x[1]) #ordinamento decrescente per la LMI
    lmi= OrdinamentoLMI[:20] #ottengo i 20 bigrammi (bigramma, LMI)
    
 
    return lmi, Pcond, bigramsoggagg


#estrarre le frasi con 6<tokens<25 che occorre almeno due volte nel corpus di riferimento

def almenodue(frasi, listaToken):
    #inizializzo la lista che contiene le frasi
    listaFrasi = []
    #scorro la lista una frase alla volta
    for frase in frasi:
        #tokenizzo la frase
        tokens = nltk.word_tokenize(frase)
        #controllo se la lunghezza della frase in termini di token corrisponde ad un numero compreso tra 6 e 25
        if len(tokens)>6 and len(tokens)<25:
            #scorro la lista un token alla volta
            for tok in tokens:
                #calcolo la frequenza del token
                freqTok = listaToken.count(tok)
                #controllo se ogni token ha frequenza maggiore di 2
            if freqTok>2:
                    #aggiungo la frase alla lista
                     
             listaFrasi.append(frase)
    #restituisco i risultati
    return listaFrasi

def Calcola0(frase, distribuzioneTok, numeroToken):
    #scorro la lista un token alla volta
    for tok in frase:
        #calcolo la probabilita del token con la distribuzione di frequenza
        probTok = (float(distribuzioneTok[tok]))/(numeroToken)
    #restituisco il risultato
    return probTok



def CalcolaMarkov2(frase, distribuzioneTok, numeroToken, bigrammiTokPos, probTok):
    #calcolo la distribuzione di frequenza dei bigrammi
    distribuzioneBig = nltk.FreqDist(bigrammiTokPos)
    #scorro la lista un token alla volta
    for tok in frase:
        #calcolo la probabilita del token con la distribuzione di frequenza
         
        #divido la frase in bigrammi
        bigrammiFrase = list(bigrams(frase))
            #scorro la lista un bigramma alla volta
        for bigramma in bigrammiFrase:
                #calcolo la probabilita del bigramma con la distribuzione di frequenza
                probBig = (float(distribuzioneBig[bigramma]))/(numeroToken)
                probig = (float(distribuzioneBig[bigramma[0]]))/(numeroToken)
                #calcolo la probabilita della frase tramite un modello di Markov di ordine 2
                probabilita = (float(probBig)/(probTok)) * (float(probig)/(probTok))
    #restituisco il risultato
    return probabilita




def CalcolaProbFrasi(listaFrasi, distribuzioneTok, numeroToken, bigrammiTokPos, probTok):
    #inizializzo le variabili per il calcolo delle probabilita delle frasi
    probMax0 = 0
    probMax1 = 0
    fraseProbMax0 = ""
    fraseProbMax1 = ""
    #scorro la lista una frase alla volta
    for frase0 in listaFrasi:
        #calcolo la probabilita della frase con un modello di Markov di ordine 0
        prob0 = Calcola0(frase0, distribuzioneTok, numeroToken)
        #controllo se la frase ha probabilita maggiore di probMax0
        if prob0>probMax0:
            #assegno alla variabile probMax0 il valore della probabilita della frase
            probMax0 = prob0
            #assegno alla variabile fraseProbMax0 la frase con probabilita maggiore
            fraseProbMax0 = frase0
    #stampo la frase con probabilita maggiore 
    print ("Frase: '", fraseProbMax0, "' Probabilità:", probMax0, ".")
    print ("\n")
    print ("\n")
#calcolo la probabilita con un modello di Markov di ordine 2
    for frase1 in listaFrasi:

        prob1 = CalcolaMarkov2(frase1, distribuzioneTok, numeroToken, bigrammiTokPos, probTok)
        #controllo se la frase ha probabilita massima
        if prob1>probMax1:
                #assegno alla variabile probMax1 il valore della probabilita della frase
           probMax1 = prob1
                  #assegno alla variabile fraseProbMax1 la frase con probabilita maggiore
           fraseProbMax1 = frase1
    #stampo la frase con probabilita maggiore secondo il modello di Markov di ordine 0
    print ("Frase con probabilità piu alta calcolata attraverso un modello di Markov di ordine 2:")
    print ("Frase: '", fraseProbMax1, "' Probabilità:", probMax1, ".")
 
    print ("\n")
    print ("\n")

    return probMax0






"""dopo aver individuato e classificato le Entità Nominate (NE) presenti nel testo, estraete:
◦ i 15 nomi propri di persona più frequenti, ordinati per frequenza;"""

def Nomi(tokenPOSTot):

    nomitag = nltk.ne_chunk(tokenPOSTot)
    persone = []
    for nodo in nomitag:
        Nomi = ""
        #se il nodo ha l'attributo "label"
        if hasattr(nodo, "label"):
            if nodo.label() in ["PERSON"]:
                #se il valore dell'attributo corrisponde a una persona li inserisce in una lista apposita
                for pNomi in nodo.leaves():
                    Nomi = Nomi + " " + pNomi[0]
                if (nodo.label() == "PERSON"):
                    persone.append(Nomi)

    personePiuFrequenti = nltk.FreqDist(persone).most_common(15)
    return personePiuFrequenti





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
    #calcolo le 10 POS piu' frequenti
    PoSfreq1 = PoSfreq(tokenPOSTot1)
    PoSfreq2 = PoSfreq(tokenPOSTot2)    

#Calcolo i trigrammi
    bigrammi1 = list(bigrams(tokenPOSTot1))
    bigrammi2 = list(bigrams(tokenPOSTot2))
#calcolo i 10 trigrammi di POS piu' frequenti
    bigrammifreq1 = Calcolabigrammi(bigrammi1)
    bigrammifreq2 = Calcolabigrammi(bigrammi2)

#Calcolo i trigrammi
    trigrammi1 = list(trigrams(tokenPOSTot1))
    trigrammi2 = list(trigrams(tokenPOSTot2))
#calcolo i 10 trigrammi di POS piu' frequenti
    trigrammifreq1 = Calcolatrigrammi(trigrammi1)
    trigrammifreq2 = Calcolatrigrammi(trigrammi2)
 #calcolo i 20 aggettivi piu' frequenti
    aggettivifre1 = aggettivi(tokenPOSTot1)
    aggettivifre2 = aggettivi(tokenPOSTot2)
    
    #calcolo i 20 verbi piu' frequenti
    avverbifre1 = avverbi(tokenPOSTot1)
    avverbifre2 = avverbi(tokenPOSTot2)
    
    lmi_1, Pcond_1, bigramsoggagg_1 = listaSostagg(tokenPOSTot1, listaToken1)
    lmi_2, Pcond_2, bigramsoggagg_2 = listaSostagg(tokenPOSTot2, listaToken2)

    #listaProbCond_sorted_20_1, listaLMI_sorted_20_1 , bigrammifrequenti_1 = bigrammi3(listaToken1, tokenPOSTot1)
    #listaProbCond_sorted_20_2, listaLMI_sorted_20_2 , bigrammifrequenti_2  = bigrammi3(listaToken2, tokenPOSTot2)
    
   
#3 punto
    sentence1 = almenodue(frasi1, listaToken1)
    sentence2 = almenodue(frasi2, listaToken2)

    distribuzioneTok1 = nltk.FreqDist(listaToken1)
    distribuzioneTok2 = nltk.FreqDist(listaToken2)

    bigrammiTokPOS1 = list(bigrams(sentence1))
    bigrammiTokPOS2 = list(bigrams(sentence2))

    n1= 0.0415282392026578
    n2= 0.034578573633837

#4 punto
    nomiPiuFrequenti1 = Nomi(tokenPOSTot1)
    nomiPiuFrequenti2 = Nomi(tokenPOSTot2)




    print ("\n")
    print("\t***___PROGRAMMA 2 Isidoro Allegretti 559896___***")
    print ("\n")
    print ("\t___Punto 1___")
    print ("\n")
#le 10 PoS più frequenti
    print ("10 PoS più frequenti di", file1)
    frequenze(PoSfreq1)
    print ("\n")
    print ("10 PoS più frequenti di", file2)
    frequenze(PoSfreq2)
    print ("\n")
    print ("\n")

#10 biigrammi di PoS più frequenti
    print ("10 bigrammi di POS piu' frequenti di", file1)
    frequenzabigrammi(bigrammifreq1)
    print ("\n")
    print ("10 bigrammi di POS piu' frequenti di", file2)
    frequenzabigrammi(bigrammifreq2)
    print ("\n")
    print ("\n")

#10 trigrammi di PoS più frequenti
    print ("10 trigrammi di POS piu' frequenti di", file1)
    frequenzatrigrammi(trigrammifreq1)
    print ("\n")
    print ("10 trigrammi di POS piu' frequenti di", file2)
    frequenzatrigrammi(trigrammifreq2)
    print ("\n")
    print ("\n")

 #20 aggettivi più frequenti
    print ("20 aggettivi piu' frequenti di", file1)
    stampafreq(aggettivifre1)
    print ("\n")
    print ("20 aggettivi piu' frequenti di", file2)
    stampafreq(aggettivifre2)
    print ("\n")
    print ("\n")
 #20 avverbi pìù frequenti   
    print ("20 avverbi piu' frequenti di", file1)
    stampafreq(avverbifre1)
    print ("\n")
    print ("20 avverbi piu' frequenti di", file2)
    stampafreq(avverbifre2)
    print ("\n")
    print ("\n")
    
    print ("\t___Punto 2___")
    print ("\n")

    
#estrarre e ordinare i 20 bigrammi composti da aggettivo e sostantito dove ogni token ha frequenza > 3
    
    print("20 bigrammi composti da aggettivo e sostantivo con frequenza > 3 in", file1)
    for i in bigramsoggagg_1:
        print("\tIl bigramma", i[0], "con frequenza", i[1])
    print ("\n")
    print("20 bigrammi composti da aggettivo e sostantivo con frequenza > 3", file2)
    for i in bigramsoggagg_2:
        print("\tIl bigramma", i[0], "con frequenza", i[1])
    print ("\n")
    print ("\n")
    
    print("con probabilità condizionata massima in", file1)
    for i in Pcond_1:
        print("\t", i[0], "con  probabilità", i[1])
    print ("\n")
    print("con probabilità condizionata massima in", file2)
    for i in Pcond_2:
        print("\t", i[0], "con probabilità", i[1])
    print ("\n")
    print ("\n")


    print("con forza associativa (LMI) massima in", file1)
    for i in lmi_1:
        print("\t", i[0], "con forza associativa", i[1])
    print ("\n")
    print("con forza associativa (LMI) massima in", file2)
    for i in lmi_2:
        print("\t", i[0], "con forza associativa", i[1])
    print ("\n")
    print ("\n")

    print ("\t___Punto 3___")
    print ("\n")


    
    print ("distribuzione di frequenza dove ogni token 6<token<25 e ha una frequenza maggiore di 2.")
    print ("\n")
    print ("file", file1)
    CalcolaProbFrasi(sentence1, distribuzioneTok1, ntoken1, bigrammiTokPOS1, n1)
    print ("\n")
    print ("distribuzione di frequenza dove ogni token 6<token<25 e ha una frequenza maggiore di 2.")
    print ("file:", file2)
    CalcolaProbFrasi(sentence2, distribuzioneTok2, ntoken2, bigrammiTokPOS2, n2)
    print ("\n")
    print ("\n")
    



    print ("\t___Punto 4___")
    print ("\n")
    print("Quindici nomi propri di persona più frequenti in", file1)
    for i in nomiPiuFrequenti1:
        print("\tIl nome", i[0], "con frequenza", i[1])
    print ("\n")
    print("Quindici nomi propri di persona più frequenti in", file2)
    for i in nomiPiuFrequenti2:
        print("\tIl nome", i[0], "con frequenza", i[1])

    

main(sys.argv[1], sys.argv[2])
