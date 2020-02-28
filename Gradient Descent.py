# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
import os
os.chdir("/Users/Alex2/Desktop/Fichiers") # à modifier...



# variables globales :
#-----------------------------------------------------------------------------------------------


N=500 # le nombre de lignes qu'on va lire (par défaut : 501)
t = 1 # la taille des groupes (par défaut : 1) Attention : il faut t|N...


#-----------------------------------------------------------------------------------------------
# calcul du gradient selon la i^e variable :






def gradi(f, A, i, epsilon):
    """ prend en argument une fonction f et un vecteur A de Rn, retourne la dérivée ârtielle de de f  par rpport à i en A"""
    X = np.copy(A) # Pour ne pas modifier le vecteur A
    Xeps = np.copy(X)
    Xeps[i,0] += epsilon
    return (f(Xeps)-f(X))/epsilon # Expression de la dérivée : lim(e tend vers 0) du taux d'accroissement

def grad(f,A,epsilon):
    """prend en argument une fonction f, un vecteur A et retourne le gradient de f en A"""
    N,_ = np.shape(A)
    G = np.zeros((N,1)) # G est le vecteur gradient de f en A
    for i in range(N):
        G[i,0] = gradi(f,A,i,epsilon)
    return G




#-----------------------------------------------------------------------------------------------
# Normes, distances euclidienne: 




def norme(J):
    return np.trace(np.dot(J,np.transpose(J))) # norme euclidienne

def distance(A,B):
    return norme(A-B)


def evalue(A,D,X):
    return distance(A,np.dot(D,X)) # Distance entre A, fixée et un vecteur DX qui est égal au produit D(matrice dictionnaire) et X(l'inconnue du problème)



def teste(X):
    """ Prend en argument un vecteur X de Rn et retourne la distance euclidienne de A à DX"""
    return evalue(A,D,X)


def adapte(X): # pour que la somme soit 1 :
    """Prend en argument un vecteur X et le le normalise"""
    N,_ = np.shape(X)
    somme = 0
    for i in range(N):
        somme += X[i,0]
    return X/somme



#-----------------------------------------------------------------------------------------------
# Descente du Gradient :



def dg(A,D,tg):
    """Prend en argument un vecteur A (vecteur observation) et une matrice dictionnaire D et renvoie la solution du problème de minimisation : retourne le vecteur X tel que la distance A-DX soit minimale""" 
    L,N = np.shape(D) #Dimensions de la matrice dictionnaire
    X = np.zeros((N,1))
    for i in range(N):
        X[i] = 1/N
    c = 1/norme(D)
    epsilon = 0.001 #Permet de régler la précision sur le calcul du gradient
    nb_max_iterations = 100 #Eviter que l'algorithme ne tourne trop longtemps 
    compteur = 0
    precision = 0.01 #
    norme_actu = evalue(A,D,X)
    while norme_actu > precision and compteur < nb_max_iterations :
        compteur += 1
        X -= c * grad(teste,X,epsilon)
        tmp = norme_actu
        norme_actu = evalue(A,D,X)
        if tmp == norme_actu:
            return adapte(X),compteur,tmp
        if compteur%10 == 0 and not tg:
            print(X,norme_actu,compteur,"\n\n")
    return adapte(X),compteur,evalue(A,D,X)




## Résultats de l'expérience pour tester si ça marche :


def lire(fichier):
    """Extrait les données du fichier texte de manière à pouvoir les traiter"""
    res = np.array([[0] for i in range(N//t)],float)
    fichier = open(fichier,'r') # Ouverture du fichier en mode lecture seule
    contenu = fichier.read()
    compteur = 0
    for i in range(N):
        tmp = []
        for j in range(12):
            ligne = ""
            while(contenu[compteur] != "\n"):
                ligne += str(contenu[compteur])
                compteur += 1
            compteur += 1
            tmp += ligne
            if j == 9:
                res[i//t,0] += float(ligne) # parce qu'il prend que des entiers...
    fichier.close()
    return res

#-----------------------------------------------------------------------------------------------
# Elements seuls : Les spectres sont acquis avec le spectro du lycée

BP_1 = lire("1_bp.txt")
CU_2 = lire("2_cu.txt")
CU_3 = lire("3_cu.txt")
KMNO_4 = lire("4_kmno.txt")
MN0_5 = lire("5_mno.txt")
MNO_6 = lire("6_mno.txt")

#-----------------------------------------------------------------------------------------------
# Mélanges : (numero_elt1/volume_elt2/volume) Spectres acquis avec notre spectro

A_15_210 = lire("A.txt")
B_110_25 = lire("B.txt")
C_25_510 = lire("C.txt")
D_210_55 = lire("D.txt")
E_2id_5id = lire("E.txt")
F_1id_2id = lire("F.txt")

# on simplifie les notations...
e1=BP_1
e2=CU_2
e3=CU_3
e4=KMNO_4
e5=MN0_5
e6= MNO_6

#-----------------------------------------------------------------------------------------------
# Mélanges expérimentaux : Listes contenant les informations de l'expérience : Spectre du mélange, matrice dictionnaire, proportions réelles des solutions

mA = (A_15_210,e1,e2,(5,10),"Bleu patenté 10mL et Cu2+ 5mL") 
mB = (B_110_25,e1,e2,(10,5),"Bleu patenté 5mL et Cu2+ 10mL")
mC = (C_25_510,e2,e5,(5,10),"Cu2+ 5mL et MnO42- 10mL")
mD = (D_210_55,e2,e5,(10,5),"Cu2+ 5mL et MnO42- 10mL")
mE = (E_2id_5id,e2,e5,(1,1),"Cu2+ et MnO42- en proportions égales")
mF = (F_1id_2id,e1,e2,(1,1),"Bleu patenté et Cu2+ en proportions égales")




# A = DX

def retrouve(M,rienafficher=True,tg=True): # Mélange M
    """Trouve le minimum X de la fonction f(X) = norme(A-DX) avec l'algorithme de descente du gradient"""
    global A #Valeurs du spectre d'absorance mesurées par notre spectro
    global D #Matrice dictionnaire (valeurs mesurées avec le spectro du lycée
    A = M[0]
    x,y = M[1],M[2]
    D = np.concatenate((x,y),axis=1)
    X,n_etapes,precision = dg(A,D,tg) #Descente du gradient
    print("\n\n\tMélange de",M[4],":\n")
    print("On a trouvé le résultat en",n_etapes,"étapes.\nL'erreur est :",precision,"\n\n  Et le résultat est :\n",X)
    vx,vy = M[3]
    v = vx+vy
    propx = vx/v
    propy = vy/v
    print("Les proportions réelles sont :",propx,propy)
    print("\n\n")
    if not rienafficher :
        dessin(M,X,propx,propy)
    
def dessin(M,X,propx,propy):
    A = M[0]
    x,y = M[1],M[2]
    plt.plot(x,y)
    plt.draw()
    plt.show()
    tracer(x,propx)
    tracer(y,propy)
    tracer(A,1,"mel")
    tracer(propx*x+propy*y,1,"g")
    tracer(X[0,0]*x+X[1,0]*y,1,"black")
    plt.legend(("élément 1","élément 2","Mélange réel","Combinaison linéaire théorique","Combinaison linéaire retrouvée"))
    plt.grid()
    plt.title(M[4])
    plt.xlim(400, 900)
    plt.draw()
    plt.show()


def tracer(c, prop,car="elt"):
    n,_=np.shape(c)
    x = [400+(t*i) for i in range(n)]
    if car=="elt":
        y = [prop*c[i,0]/(t) for i in range(n)]
        plt.plot(x,y,'b',lw=0.3)
    elif car=="mel":
        y = [prop*c[i,0]/(t) for i in range(n)]
        plt.plot(x,y,'r',lw=1.5)
    else:
        y = [prop*c[i,0]/(t) for i in range(n)]
        plt.plot(x,y,car,lw=0.8)
    #plt.draw()
    plt.show()
            

retrouve(mA,False,True)
#retrouve(mB,False,True)
#retrouve(mC,False,True)
#retrouve(mD,False,True)
#retrouve(mE,False,True)
#retrouve(mF,False,True)












## Pour des tests :

#D = np.array([[2]])
#A = np.array([[1]])

#Y = np.array([[0.5],[0.25],[0.25]])
#D = np.array([[1,50,1],[3,1,1],[5,10,2],[13,100,5],[5,10,3],[13,10,4],[5,10,1],[13,10,8],[5,5,10],[9,13,10],[0,5,10],[13,1,10],[1,2,3],[4,5,6]]) 
#A = np.dot(D,Y)

# avec des matrices un peu plus grosses :
# L : nb de longueurs d'ondes
# N : nb d'éléments
#L = 500
#N = 100
# 
#B = np.random.rand(L,1)/10 # le bruit !
#Y = np.array([[i] for i in range(1,N+1)])
#D = np.random.rand(L,N)
#A = np.dot(D,Y) #+ B
#
#
#
#print(dg(A,D))
#
#







