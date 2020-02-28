#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
# Imports :

import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir("/Users/Alex2/Desktop/Fichiers")

fichier = open("Résultat5_mno.txt", "r",encoding = 'UTF-8')
f = fichier.readlines()

titre = input('Entrer le titre :')

n = len(f)

# Séparer les abscisses (lambda) et les ordonnées (A)
c = 0
while f[c] != 'A\n' :
    c += 1
print(c)


# Construire les listes à partir des fichiers :

X = []
Y = []

for k in range(c+1,n) :
    if float(f[k]) <0 :
        Y.append(0)
    else :
        Y.append(float(f[k]))
     
for k in range(1, c) :
    X.append(int(f[k]))
     
#-----------------------------------------------------------------------------------------------
# Spectres à traiter : Il s'agit de rentrer la liste L qui est échantillonnée par la suite

L = [X, Y]



#-----------------------------------------------------------------------------------------------#-----------------------------------------------------------------------------------------------
# Fonction de découpage sur n intervalles : 


def decoupage(L, N) :
    """ Prend comme argument une liste de taille 2 (une liste d'ascisses et une liste d'ordonnées) et renvoie N listes de taille n/N. Cela correspond à une subdivision sur N intervalles d'une courbe""" 
    n = len(L[0])
    p = n/(N)
    
    Ltot = [] # liste de liste à renvoyer à la fin
    
    for k in range(N) :
        A = np.zeros((2, int(n/N))) # On crée une liste vide pour acceuillir les couples de poits de la subdivision k
        for i in range(int(k*p),int((k+1)*p)):
            A[0][i-k*p] = L[0][i]
            A[1][i-k*p] = L[1][i] # Remplissage de cette liste vide avec les valeurs de L
        Ltot.append(A)
    return Ltot
    
    
    
#-----------------------------------------------------------------------------------------------
# Fonction de recherche de minimum et maximum sur un intervalle :


def extremes(I) :
    """prend en argument une liste de couples [(X,Y),...,(Xn, Yn)] et renvoie les coordonnées du max et du min"""
    n = len(I[1])
    M = I[1][0] # Max
    i = 0 # indice du max
    m = I[1][0] # Min
    j = 0 #Indice du min
    
    for k in range(n) :
        if I[1][k] > M :
            i = k
            M = I[1][k]
        
    for k in range(n) :
        if I[1][k] < m :
            j = k
            m = I[1][k]
    return([I[0][i],M], [I[0][j],m]) # retourne [(xMAX, yMAX), (xmin, ymin)]
   


#-----------------------------------------------------------------------------------------------
# Fonctionq d'échantillonnage :

def echantillonnage_brut(L,N) :
    """Prend en argument un liste de points (X,Y) avec a<X<b, on subdivise en n intervalles et retourne la liste des points extremes sur chaque intervalle"""
    P = [] # Liste des points initialement vide
    for k in range(N) :
        P.append(extremes(decoupage(L,N)[k])) # On ajoute les coordonnées des points interessants
    return P # P contient 100 couples (pt min, pt max), soit 200 couples coordonnées
    

    

def echantillonnage(L, N) :
    """ Prend comme argument une liste de taille 2 : la permière correspond aux abscisses et la deuxième aux ordonnées (mesures du capteur CCD). Elle retourne les listes de valeurs échantillonées"""
    F = echantillonnage_brut(L, N) #On échantillonne la courbe brute mesurée par le CCD
    Ye = []
    Xe = []
    for k in range(N) :
        if F[k][0][0] < F[k][1][0] : #Il faut faire attention à ranger les points avec les abscisses croissantes
        
            Ye.append(F[k][0][1])
            Ye.append(F[k][1][1])
            Xe.append(F[k][0][0])
            Xe.append(F[k][1][0])
        else : 
            Ye.append(F[k][1][1])
            Ye.append(F[k][0][1])
            Xe.append(F[k][1][0])
            Xe.append(F[k][0][0])
            
    return [Xe, Ye]
    


#-----------------------------------------------------------------------------------------------
# Fonctions de tracé de spectre : 


def graphe(L, N) :
    """ Admet une liste de taille 2 en argument et affiche le spectre à échantillonner et le spectre echantillonné"""
    A = echantillonnage(L, N)
    Xe = A[0]
    Ye = A[1]
    plt.plot(Xe, Ye, label="Spectre échantillonné " + str(N) + "points", marker = "o", color = "r" )
    plt.plot(L[0], L[1], label="Spectre à échantillonner", color = "k")
    plt.xlabel("Lambda(nm)")
    plt.ylabel("A")
    plt.legend()
    plt.show()


#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
