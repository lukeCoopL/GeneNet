#from re import M, S
from time import time
#from graphviz import Digraph
from matplotlib.pyplot import xcorr
import networkx as nx
#import itertools
import copy
#from networkx.readwrite.json_graph import tree
import numpy as np
#from networkx.algorithms.shortest_paths.unweighted import all_pairs_shortest_path, predecessor
import math
#from collections import defaultdict
from networkx.utils import py_random_state
import random as random
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import permutation_test_score
#from scipy.stats import expon as ex 
from Genetc.utilities import *
from Genetc.duplication import *
from Genetc.alignment import *
from math import comb

#-----------------------------------------------------------
#Ancestral Reconstruction
def expected_number_of_self_loops(G,G_seed,p):
  SL=len(list(nx.selfloop_edges(G_seed)))
  for i in range(len(G_seed.nodes()),len(G.nodes())):
    SL=SL+(SL/i)*(1-p)
  return SL
def expected_number_of_self_loops(G,G_seed,p):
  SL=len(list(nx.selfloop_edges(G_seed)))
  n=len(G.nodes)
  k=len(G_seed.nodes)
  fact1=1
  fact2=1
  fact3=1
  fact4=1
  for i in range(1,k):
    fact1=fact1*i
  for i in range(1,k+1):
    fact2=fact2*(i-p)
  for i in range(k,n):
    SL=SL+(SL/i)*(1-p)
  return 
def expected_number_of_isolates(G,G_seed,p,iso_number=0):
  SL=len(list(nx.selfloop_edges(G_seed)))
  deg=sum([d for (n, d) in nx.degree(G_seed)]) / len(G_seed.nodes())
  iso=len(list(nx.isolates(G_seed)))
  degList=[G_seed.degree(i) for i in G_seed.nodes]
  isoOld=iso
  for i in range(len(G_seed.nodes()),len(G.nodes())+iso_number):
    #print(range(int(np.floor(min(degList))),int(np.ceil(max(degList)))))
    #print(list(np.histogram(degList,bins=int(np.ceil(max(degList)))-int(np.floor(min(degList))))))
    
    summ=0
    for deg in degList:
      #if 0 in degList:
        #deg=random.choices(range(int(np.ceil(max(degList)))),weights=list(np.histogram(degList,bins=int(np.ceil(max(degList))),density=True)[0]))[0]
      #else:
        #deg=random.choices(range(1,int(np.ceil(max(degList)))),weights=list(np.histogram(degList,bins=int(np.ceil(max(degList)))-1,density=True)[0]))[0]
    
      summ=summ+2*(p/2)**deg
    iso=iso+(1/i)*summ
    #if iso>len([i for i in degList if i==0])+1:
    #  degList.append(0)
    #else:
    if 0 in degList:
        deg=random.choices(range(int(np.ceil(max(degList)))),weights=list(np.histogram(degList,bins=int(np.ceil(max(degList))),density=True)[0]))[0]
    else:
        deg=random.choices(range(1,int(np.ceil(max(degList)))),weights=list(np.histogram(degList,bins=int(np.ceil(max(degList)))-1,density=True)[0]))[0]
    for i in range(int((1-p)*deg)):
      ind=random.choices(range(len(degList)),weights=degList)[0]
      degList[ind]=degList[ind]+1
      if degList[ind]<0:
        degList[ind]=0
    degList.append(deg*(1-p/2))
    ind=random.choices(range(len(degList)))[0]
    degList[ind]=degList[ind]-(p/2)*deg
    if degList[ind]<0:
      degList[ind]=0
      
  print(degList)
    #deg=deg*(1+(1-2*p)/(i+1))+(4*(1-p)*(SL/i))/(i+1)
    #SL=SL+(SL/i)*(1-p)
    
  return iso
def expected_number_of_isolates_test(G,G_seed,p,iso_number=0):
  iso=len(list(nx.isolates(G_seed)))
  ones=len([n for (n,d) in G_seed.degree() if d==1])
  SL=len(list(nx.selfloop_edges(G_seed)))
  deg=sum([d for (n, d) in nx.degree(G_seed)]) / len(G_seed.nodes())
  effDeg=deg
  realDeg=deg
  print(deg)
  numNodes=int(len(G_seed.nodes())-iso)
  counter=numNodes
  while counter<len(G.nodes())+iso_number:
    counter=counter+1
    #prevNumNodes=numNodes
    realDeg=realDeg*(1+(1-2*p)/(counter))+(2*(1-p)*(SL/counter-1))/(counter)
    isoProb=iso/(counter)+((counter-iso-SL-ones)/counter)*(p/2)**effDeg+ones/counter*(p/2)
    
    rando=random.random()
    if rando<isoProb:
      iso=iso+1
    #else:
    #  numNodes=numNodes+1
    
    #effDeg=effDeg*(1+(1-2*p)/(numNodes))+(2*(1-p)*(SL/prevNumNodes))/(numNodes)
    SL=SL+(SL/counter)*(1-p)
  return iso
def FFL_clustering_recurrence_no_selfloop_no_r(G,G_seed,p):
  clust=average_FFL_triangles(G_seed)
  for i in range(len(G_seed.nodes()),len(G.nodes())):
    clust=(1/(i+1))*(i*clust+(3*(1-p/2)**2)*clust-3*(p-p**2/4)*clust)
  return clust
def degree_recurrence(G,G_seed,p,r,iso_number=0):
  iso=len(list(nx.isolates(G_seed)))
  SL=len(list(nx.selfloop_edges(G_seed)))
  deg=sum([d for (n, d) in nx.degree(G_seed)]) / len(G_seed.nodes())
  print(deg)
  for i in range(len(G_seed.nodes()),len(G.nodes())+iso_number):
    summ=0
    
    deg=deg*(1+(1-2*p)/(i+1)-(2*r)/(i*(i+1)))+(4*(1-p)*(SL/i))/(i+1)+(2*r)/(i+1)
    SL=SL+(SL/i)*(1-p)
  
  return ((i+1)*deg)/(i+1-iso_number)
def degree_recurrence_test(G,G_seed,p,iso_number=0):
  iso=len(list(nx.isolates(G_seed)))
  SL=len(list(nx.selfloop_edges(G_seed)))
  deg=sum([d for (n, d) in nx.degree(G_seed)]) / len(G_seed.nodes())
  effDeg=deg
  realDeg=deg
  print(deg)
  numNodes=int(len(G_seed.nodes())-iso)
  counter=numNodes
  while numNodes<len(G.nodes())+iso_number:
    counter=counter+1
    prevNumNodes=numNodes
    realDeg=realDeg*(1+(1-2*p)/(counter))+(2*(1-p)*(SL/counter-1))/(counter)
    isoProb=iso/(counter)+((counter-iso-SL)/counter)*(p/2)**effDeg
    
    rando=random.random()
    if rando<isoProb:
      iso=iso+1
    else:
      numNodes=numNodes+1
    
    effDeg=effDeg*(1+(1-2*p)/(numNodes))+(2*(1-p)*(SL/prevNumNodes))/(numNodes)
    SL=SL+(SL/counter)*(1-p)
    if numNodes<1:
      break
    #if int(numNodes)%500==0:
    #  print(numNodes)
    
  
  return effDeg

def list_based_expected_degree(G,G_seed,p):
  degList=[G_seed.degree[i] for i in G_seed.nodes()]
  step=len(G_seed.nodes)
  while step<len(G.nodes()):
    picked=random.sample(degList,k=1)
    picked=picked[0]
    while picked==0:
      picked=random.sample(degList,k=1)
      picked=picked[0]
    
    delNumber=int(p*picked)
    pickDex=degList.index(picked)
    for i in range(delNumber):
      rando=int(random.random()*len(degList))
      while rando==pickDex:
        rando=int(random.random()*len(degList))
        degList[rando]=degList[rando]-1
    
    degList[pickDex]=degList[pickDex]-int(delNumber/2)
    degList.append(int(delNumber/2))
    step=step+1
  
  return np.mean(degList)

def fit_p_degree_recurrence(G,G_seed,eps,r,iso_number=0):
  trueDeg=sum([d for (n, d) in nx.degree(G)]) / len(G.nodes())
  degMax=degree_recurrence(G,G_seed,0,r,iso_number=int(0*iso_number))
  degMin=degree_recurrence(G,G_seed,1,r,iso_number=int(1*iso_number))
  print(degMin,degMax,trueDeg)
  if degMin>trueDeg or degMax<trueDeg:
    raise ValueError('Degree recurrence does not fit')
  pMin=1

  pMax=0
  while pMin-pMax>eps:
    pDash=(pMin+pMax)/2
    deg=degree_recurrence(G,G_seed,pDash,r,iso_number=int(pDash*iso_number))
    if deg<trueDeg:
      pMin=pDash
    else:
      pMax=pDash
  return pMin

def adj_distance(G1,G2):
  if len(G1.nodes)!=len(G2.nodes):
    raise ValueError('Graphs are unequal size')
  A1=nx.to_numpy_matrix(G1)
  A2=nx.to_numpy_matrix(G2)
  E1=list(np.linalg.eigvals(A1))
  E2=list(np.linalg.eigvals(A2))
  E1=np.sort(E1)
  E2=np.sort(E2)
  for i in range(len(E1)):
    summ=summ+(E1[i]-E2[i])**2
  summ=np.sqrt(summ)
  return summ
def kendalls_tau(A_pred,A_true,count_all=False,useSeedVector=False,seedVector=None,orderedSeed=False):
  if count_all==False:
    if len(A_pred)!=len(A_true):
      raise ValueError('Vectors are unequal length')
  nd=0
  nc=0
  for ind,i in enumerate(A_pred):
    for jnd,j in enumerate(A_pred):
      if jnd>ind:
        if useSeedVector==False:
          if i not in A_true or j not in A_true:
            nd=nd+1
          elif A_true.index(i)<A_true.index(j):
            nc=nc+1
          else:
            nd=nd+1
        else:
          if orderedSeed:
              if i in A_true and j not in A_true and j in seedVector:
                nc=nc+1
              elif i not in A_true and j in A_true and i in seedVector:
                nd=nd+1
              elif (i not in A_true and j not in A_true) and (i in seedVector and j in seedVector) and seedVector.index(i)>seedVector.index(j):
                nc=nc+1
              elif (i not in A_true and j not in A_true) and (i in seedVector and j in seedVector) and seedVector.index(i)<seedVector.index(j):
                nd=nd+1
              elif i not in A_true or j not in A_true:
                print("noone be here")
                nd=nd+1
              elif A_true.index(i)<A_true.index(j):
                nc=nc+1
              else:
                nd=nd+1
          else:  
            if i in A_true and j not in A_true and j in seedVector:
              nc=nc+1
            elif i not in A_true and j in A_true and i in seedVector:
              nd=nd+1
            elif i not in A_true or j not in A_true:
              nd=nd+1
            elif A_true.index(i)<A_true.index(j):
              nc=nc+1
            else:
              nd=nd+1

  #nc=len(set(A)&set(B))
  #nd=len(set(A)^set(B))
  n=len(A_pred)
  return (nc-nd)/(comb(n,2))

def spearmans_footrule(A_pred,A_true):
  if len(A_true)!=len(A_pred):
    raise ValueError('Vectors are unequal length')
  A_true_dict=dict()
  A_pred_dict=dict()
  summ=0
  n=len(set(A_true))
  for ind,i in enumerate(A_true):
    A_true_dict[i]=ind
  for ind,i in enumerate(A_pred):
    A_pred_dict[i]=ind
  for i in A_true:
    summ=summ+np.abs(A_true_dict[i]-A_pred_dict[i])
  return 1-summ/(np.floor(n**2/2))

def ancestral_likelihood_dmc(G,i,j,qMod,qCon):
  
  prod=1
  anchorNeighboursIn = list(G.predecessors(i))
  anchorNeighboursOut = list(G.successors(i))
  duplicateNeighboursIn=list(G.predecessors(j))
  duplicateNeighboursOut=list(G.successors(j))
  intersectingNeighboursIn = list(set(anchorNeighboursIn)&set(duplicateNeighboursIn))
  intersectingNeighboursOut = list(set(anchorNeighboursOut)&set(duplicateNeighboursOut))
  intersectingNeighbours=intersectingNeighboursIn+intersectingNeighboursOut
  
  symmdiffNeighboursIn = list(set(anchorNeighboursIn)^set(duplicateNeighboursIn))
  symmdiffNeighboursOut = list(set(anchorNeighboursOut)^set(duplicateNeighboursOut))
  symmdiffNeighbours=symmdiffNeighboursIn+symmdiffNeighboursOut
  neighboursTot=len(intersectingNeighbours+symmdiffNeighbours)
  for k in intersectingNeighbours:
    if k!=i and k!=j:
      prod = prod*(1-qMod)
  for k in symmdiffNeighbours:
    if k!=i and k!=j:
      prod = prod*qMod/2
  if (i,j) in list(G.edges):
    prod=prod*qCon
  else:
    prod=prod*(1-qCon)
  if (j,i) in list(G.edges):
    prod=prod*qCon
  else:
    prod=prod*(1-qCon)
  #prod=prod/len(list(G.nodes))
  #prod=prod*neighboursTot
  return prod

def check_faulty_edges(G,GPrev,i,j):
  #Big source of slowness!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    prevEdgeSet=set(GPrev.edges()).difference(set(G.edges))
    #problemEdgeList= [(m,n) for (m,n) in prevEdgeSet if (m!=i and n!=i) or (m==i and n!=i) or (m!=i and n==i)]
  
    for k in prevEdgeSet:
      #if k[0]!=j and k[1]!=j:
        if k[0]!=i and k[1]!=i:
          #print("t")
          return 0
        if k[1]==i and k[0]!= i:
          if (k[0],j) not in set(G.edges):
            #print("t")
            return 0
        if k[0]==i and k[1]!= i:
          if (j,k[1]) not in set(G.edges):
            #print("t")
            return 0

def check_self_edge_condition(G,GPrev,i,j):
  #Check if the true edge set is a subset of the duplicated edge set
    if j in set(list(G.predecessors(j))) and i not in set(list(GPrev.predecessors(i))):
      #print("t")
      return 0
    else:
      
      if len(set(list(G.predecessors(j))).difference(set(list(GPrev.predecessors(i)))))!=0 and set(list(G.predecessors(j))).difference(set(list(GPrev.predecessors(i))))!={j}:
        #print("t")
        return 0
    if j in set(list(G.successors(j))) and i not in set(list(GPrev.successors(i))):
      #print("t")
      return 0
    else:
      if len(set(list(G.successors(j))).difference(set(list(GPrev.successors(i)))))!=0 and set(list(G.successors(j))).difference(set(list(GPrev.successors(i))))!={j}:
        #print("t")
        return 0

def ancestral_likelihood_of_anchor_PD(G,GPrev,i,j,q,divide_by_n=True,NK=False,timesTwo=True,non_specific=False):
  if check_self_edge_condition(G,GPrev,i,j)==0:    
    return 0
  if check_faulty_edges(G,GPrev,i,j)==0:
    return 0
    
  ''' 
    for k in set(list(G.predecessors(j))):
      if k==j:
        if i not in set(list(GPrev.predecessors(i))):
          return 0
      else:
        if k not in set(list(GPrev.predecessors(i))):
          return 0
    
    for k in set(list(G.successors(j))):
      if k==j:
        if i not in set(list(GPrev.successors(i))):
          return 0
      else:
        if k not in set(list(GPrev.successors(i))):
          return 0
  '''
    

    
  '''for k in set(GPrev.edges()):
    if k not in set(G.edges):
      if k[0]!=i and k[0]!=j and k[1]!=i and k[1]!=j:
        return 0
      if k[1]==i and k[0]!= i:
        if (k[0],j) not in set(G.edges):
          return 0
      if k[0]==i and k[1]!= i:
        if (j,k[1]) not in set(G.edges):
          return 0
  '''
  prod=1

  numNodes=len(list(G.nodes))
  anchorNeighboursIn = list(G.predecessors(i))
  anchorNeighboursOut = list(G.successors(i))
  duplicateNeighboursIn=list(G.predecessors(j))
  duplicateNeighboursOut=list(G.successors(j))
  intersectingNeighboursIn = list(set(anchorNeighboursIn)&set(duplicateNeighboursIn))
  intersectingNeighboursOut = list(set(anchorNeighboursOut)&set(duplicateNeighboursOut))
  intersectingNeighbours=intersectingNeighboursIn+intersectingNeighboursOut
  if i in intersectingNeighbours:
    intersectingNeighbours.remove(i)
  if j in intersectingNeighbours:
    intersectingNeighbours.remove(j)
  
  symmdiffNeighboursIn = list(set(anchorNeighboursIn)^set(duplicateNeighboursIn))
  symmdiffNeighboursOut = list(set(anchorNeighboursOut)^set(duplicateNeighboursOut))
  symmdiffNeighbours=symmdiffNeighboursIn+symmdiffNeighboursOut
  if i in symmdiffNeighbours:
    symmdiffNeighbours.remove(i)
  if j in symmdiffNeighbours:
    symmdiffNeighbours.remove(j)
  if (i,i) in set(GPrev.edges):  
    ###Pair the self edge with the self edge
    if (i,i) in set(G.edges):
      if (j,j) in set(G.edges):
        intersectingNeighbours.append(i)
      else:
        symmdiffNeighbours.append(i)
    elif (j,j) in set(G.edges):
        symmdiffNeighbours.append(i)
    else:
      
      return 0
    if (i,j) in set(G.edges):
      if (j,i) in set(G.edges):
        intersectingNeighbours.append(j)
      else:
        symmdiffNeighbours.append(j)
    elif (j,i) in set(G.edges):
        symmdiffNeighbours.append(j)
    else:
      
      return 0
  ###Pair the cross edge with self edge
  '''  if (i,i) in list(G.edges):
      if (j,i) in list(G.edges):
        intersectingNeighbours.append(i)
      else:
        symmdiffNeighbours.append(i)
    elif (i,i) not in list(G.edges) and (j,i) not in list(G.edges):
      return 0
    if (j,j) in list(G.edges):
      if (i,j) in list(G.edges):
        intersectingNeighbours.append(j)
      else:
        symmdiffNeighbours.append(j)
    elif (j,j) not in list(G.edges) and (i,j) not in list(G.edges):
      return 0'''
  
  intersectNo=len(intersectingNeighbours)
  symmNo=len(symmdiffNeighbours)
  
  
  prod = prod*(1-q)**intersectNo
  prod=prod*(q/2)**symmNo
  if divide_by_n:
    prod=prod/(numNodes-1)
  if non_specific==False and timesTwo ==True and NK==False and symmNo!=0:
    prod=prod*2
  if non_specific==True:
    prod=prod*2**symmNo*math.comb(symmNo+intersectNo,symmNo)
  #print("i",i,"j",j,"prod",prod,"q",q,"intersectNo",intersectNo,"symmNo",symmNo,"numNodes",numNodes)
  return prod


def most_likely_previous_graph_fast(G,q,divide_by_n=True,timesTwo=True,auto=True,non_specific=False):
  prevVec=dict()
  autoVecPair=dict()
  autoVecGraph=dict()
  likelihood=dict()
  baseLikelihood=dict()
  first=True
  for j in G.nodes():
    prevVec[j]=dict()
    autoVecPair[j]=dict()
    autoVecGraph[j]=dict()
    likelihood[j]=dict()
    baseLikelihood[j]=dict()

  
  for jnd, j in enumerate(G.nodes):
    #first=True
    for ind,i in enumerate(list(G.nodes)):
    #for ind,i in enumerate(G.nodes):
      if i!=j:
        #print("dup",j,"anc",i)
        
        done=False
        GPrev=node_merger(G,i,j,self_loops=True)
        #GPrevRev=node_merger(G,j,i,self_loops=True)
        prevVec[j][i]=GPrev
        
        
        if ind!=jnd:
          baseLikelihood[j][i]=ancestral_likelihood_of_anchor_PD(G,GPrev,i,j,q,divide_by_n,timesTwo=timesTwo,non_specific=non_specific)
          #ancestral_likelihood_of_graph_PD(G,GPrev,i,j,q,divide_by_n,timesTwo=timesTwo)
          #baseLikelihood[i][j]=baseLikelihood[j][i]
          likelihood[j][i]=baseLikelihood[j][i]
          #likelihood[i][j]=baseLikelihood[i][j]
        if first==True:
          autoVecPair[j][i]=[]
          autoVecGraph[j][i]=[]
          autoVecPair[j][i].append((j,i))
          #autoVecPair[j][i].append((i,j))
          autoVecGraph[j][i].append(GPrev)
          #autoVecGraph[j][i].append(GPrevRev)
          
          first=False
        else:
          #for jtnd, jt in enumerate(list(G.nodes)[0:jnd]):
          #for jt in [i,j]:
            
            for knd,k in enumerate(autoVecGraph[j].keys()):
                if done ==False and ped_is_automorphic(autoVecGraph[j][k][0],GPrev,k,i):
                  
                  #print("graph where",j,"merged into",i,"already found in",j,"merged into ",k)
                  autoVecGraph[j][k].append(GPrev)
                  #autoVecGraph[j][k].append(GPrevRev)
                  autoVecPair[j][k].append((j,i))
                  #autoVecPair[j][k].append((i,j))
                  done=True
                #else:
                #  print("graph where",j,"merged into",i,"NOT ISO TO",jt,"merged into ",k)
            '''if done==False:
              for jtnd, jt in enumerate(list(G.nodes)[0:jnd]):
                if i in autoVecGraph[jt].keys():
                  if done ==False and (is_automorphic_fast(autoVecGraph[jt][i][0],GPrev)):
                      print("useful")
                      #print("graph where",j,"merged into",i,"already found in",j,"merged into ",k)
                      autoVecGraph[jt][k].append(GPrev)
                      autoVecPair[jt][k].append((j,i))
                      done=True'''
            if done==False:
              #if j==4:
                #print("adding",i,"to the auto vec of",j)
              autoVecPair[j][i]=[]
              autoVecGraph[j][i]=[]
              autoVecPair[j][i].append((j,i))
              #autoVecPair[j][i].append((i,j))
              autoVecGraph[j][i].append(GPrev)
              #autoVecGraph[j][i].append(GPrevRev)
  for jnd, j in enumerate(G.nodes):
    for knd,k in enumerate(autoVecPair[j].keys()):
      for lnd,l in enumerate(autoVecPair[j][k]):
        for mnd,m in enumerate(autoVecPair[j][k]):
          if l!=m:
            #print(autoVecPair[j][k][l[0]],autoVecPair[j][k][l][1],autoVecPair[j][k][m][0],autoVecPair[j][k][m][1])
            likelihood[autoVecPair[j][k][lnd][0]][autoVecPair[j][k][lnd][1]]=likelihood[autoVecPair[j][k][lnd][0]][autoVecPair[j][k][lnd][1]]+baseLikelihood[autoVecPair[j][k][mnd][0]][autoVecPair[j][k][mnd][1]]
            #likelihood[autoVecPair[j][k][mnd][0]][autoVecPair[j][k][mnd][1]]=likelihood[autoVecPair[j][k][mnd][0]][autoVecPair[j][k][mnd][1]]+baseLikelihood[autoVecPair[j][k][lnd][0]][autoVecPair[j][k][lnd][1]]
        if auto==True:
          likelihood[autoVecPair[j][k][lnd][0]][autoVecPair[j][k][lnd][1]]=likelihood[autoVecPair[j][k][lnd][0]][autoVecPair[j][k][lnd][1]]*len(autoVecPair[j][k])
    
  '''
      #likelihood[j][i]=baseLikelihood[j][i]
  for jnd, j in enumerate(G.nodes):
    for knd,k in enumerate(prevVec[j].keys()):
      for mnd, m in enumerate(G.nodes):   
        for lnd,l in enumerate(prevVec[m].keys()):  
          if jnd<=mnd:
            if knd<lnd:
              print("jnd",jnd,"knd",knd,"mnd",mnd,"lnd",lnd)      
              #print("aaaaaaaaaaaaaaaaaaaaaaaaaa")  
              if is_automorphic_fast(prevVec[j][k],prevVec[m][l]):
                #if j==5:
                #  print("knd",knd,"lnd",lnd)  
                print("graph where",j,"merged into",k,"already found")
            
                likelihood[j][k]=likelihood[j][k]+baseLikelihood[m][l]
                likelihood[m][l]=likelihood[m][l]+baseLikelihood[j][k]'''
  '''
  for jnd, j in enumerate(G.nodes):  
    for knd,k in enumerate(prevVec[j].keys()):
      for lnd,l in enumerate(prevVec[j].keys()):  
        if knd>lnd:      
            
          if is_automorphic_fast(prevVec[j][k],prevVec[j][l]):
            #if j==5:
            #  print("knd",knd,"lnd",lnd)  
            print("graph where",j,"merged into",k,"already found")
        
            likelihood[j][k]=likelihood[j][k]+baseLikelihood[j][l]
            likelihood[j][l]=likelihood[j][l]+baseLikelihood[j][k]'''
  return likelihood

def most_likely_previous_graph_faster(G,q,divide_by_n=True,timesTwo=True,non_specific=False):
  prevVec=dict()
  autoVecPair=dict()
  autoVecGraph=dict()
  likelihood=dict()
  baseLikelihood=dict()
  autoVecSum=dict()
  first=True
  for j in G.nodes():
    prevVec[j]=dict()
    autoVecPair[j]=dict()
    autoVecGraph[j]=dict()
    likelihood[j]=dict()
    baseLikelihood[j]=dict()
    autoVecSum[j]=dict()

  
  for jnd, j in enumerate(G.nodes):
    for ind,i in enumerate(list(G.nodes)):
      if i!=j:
        done=False
        GPrev=node_merger(G,i,j,self_loops=True)
        prevVec[j][i]=GPrev
        if ind!=jnd:
          baseLikelihood[j][i]=ancestral_likelihood_of_anchor_PD(G,GPrev,i,j,q,divide_by_n,timesTwo=timesTwo,non_specific=non_specific)
         
        if first==True:
          autoVecPair[j][i]=[]
          autoVecGraph[j][i]=[]
          autoVecPair[j][i].append(i)
          autoVecSum[j][i]=baseLikelihood[j][i]
        
          autoVecGraph[j][i].append(GPrev)
         
          
          first=False
        else:
          for knd,k in enumerate(autoVecGraph[j].keys()):
            if done ==False and is_automorphic_fast(autoVecGraph[j][k][0],GPrev):
                  
              autoVecGraph[j][k].append(GPrev)
              
              autoVecPair[j][k].append(i)
              autoVecSum[j][k]=autoVecSum[j][k]+baseLikelihood[j][i]
                  
              done=True
           
            if done==False:

              autoVecPair[j][i]=[]
              autoVecGraph[j][i]=[]
              autoVecPair[j][i].append(i)
              autoVecSum[j][i]=baseLikelihood[j][i]
 
              autoVecGraph[j][i].append(GPrev)

  for j in enumerate(G.nodes):
    for k in enumerate(autoVecPair[j].keys()):
      for l in enumerate(autoVecPair[j][k]):
        likelihood[j][l]=autoVecSum[j][k]
  
  return likelihood

#------------------------------------------------------------------------------------

def ancestral_likelihood_of_graph_PD(G,GPrev,i,j,q,divide_by_n=True,timesTwo=True):
  #G=copy.deepcopy(G)
  
  #plt.figure()
  #nx.draw(GPrev,with_labels=True)
  likelihood=0
  for k in GPrev.nodes():
    
    anchorLik=ancestral_likelihood_of_anchor_PD(G,GPrev,k,j,q,divide_by_n,timesTwo=timesTwo)
    if k==1 and j=='8_1':
      print(k,j,anchorLik)
    
    likelihood=likelihood+anchorLik

  return likelihood

def most_likely_previous_graph_old(G,q,divide_by_n=True):
  prevVec=dict()
  likelihood=dict()
  baseLikelihood=dict()
  first=True
  for j in G.nodes():
    prevVec[j]=dict()
    likelihood[j]=dict()
    baseLikelihood[j]=dict()

  
  for jnd, j in enumerate(G.nodes):
    #first=True
    for ind,i in enumerate(list(G.nodes)[0:jnd]):
    #for ind,i in enumerate(G.nodes):
      if ind<jnd:
        #print("dup",j,"anc",i)
        #for j in list(G.nodes):
        #if i!=j:
        #if ind<len(list(G.nodes))-1:
        #print("Graph obtained from merging",i,"and",j)
        done=False
        GPrev=node_merger(G,i,j,self_loops=True)

        if first:
          prevVec[j][i]=GPrev
          first=False
        else:
          #if j==5:
          #  print(prevVec[j][1])
          for k in prevVec[j]:
                
            if is_automorphic_fast(GPrev,prevVec[j][k]):
              
              #print("graph where",j,"merged into",k,"already found")
                #print(i,j,"automorphic to", k,j)
                  
                #print("broken")
                #print(baseLikelihood[j][k])
              likelihood[j][k]=likelihood[j][k]+baseLikelihood[j][k]
              done=True
          if not done:
            prevVec[j][i]=GPrev
            
        if not done:
          baseLikelihood[j][i]=ancestral_likelihood_of_graph_PD(G,GPrev,i,j,q,divide_by_n)
          likelihood[j][i]=baseLikelihood[j][i]
      '''if ind>jnd:
        print("dup",j,"anc",i)
        #for j in list(G.nodes):
        #if i!=j:
        #if ind<len(list(G.nodes))-1:
        #print("Graph obtained from merging",i,"and",j)
        done=False
        GPrev=node_merger(G,j,i,self_loops=True)

        if first:
          prevVec[i][j]=GPrev
          first=False
        else:
          for k in prevVec[i]:
                
            if is_automorphic_fast(GPrev,prevVec[i][k]):
              
              print("graph where",j,"merged into",k,"already found")
                #print(i,j,"automorphic to", k,j)
                  
                #print("broken")
                #print(baseLikelihood[j][k])
              likelihood[i][k]=likelihood[i][k]+baseLikelihood[i][k]
              done=True
          if not done:
            prevVec[i][j]=GPrev
            
        if not done:
          baseLikelihood[i][j]=ancestral_likelihood_of_graph_PD(G,GPrev,j,i,q,divide_by_n)
          likelihood[i][j]=baseLikelihood[i][j]'''
  return likelihood

def most_likely_previous_graph(G,q,divide_by_n=True,timesTwo=True):
  prevVec=dict()
  autoVecPair=dict()
  autoVecGraph=dict()
  likelihood=dict()
  baseLikelihood=dict()
  first=True
  for j in G.nodes():
    prevVec[j]=dict()
    autoVecPair[j]=dict()
    autoVecGraph[j]=dict()
    likelihood[j]=dict()
    baseLikelihood[j]=dict()

  
  for jnd, j in enumerate(list(G.nodes)):
    #first=True
    for ind,i in enumerate(list(G.nodes)):
    #for ind,i in enumerate(G.nodes):
      if i!=j:
        #print("dup",j,"anc",i)
        
        done=False
        GPrev=node_merger(G,i,j,self_loops=True)
        #GPrevRev=node_merger(G,j,i,self_loops=True)
        prevVec[j][i]=GPrev
        
        
        if ind<jnd:
          baseLikelihood[j][i]=ancestral_likelihood_of_graph_PD(G,GPrev,i,j,q,divide_by_n,timesTwo=timesTwo)
          baseLikelihood[i][j]=baseLikelihood[j][i]
          likelihood[j][i]=baseLikelihood[j][i]
          likelihood[i][j]=baseLikelihood[i][j]
        if first==True:
          autoVecPair[j][i]=[]
          autoVecGraph[j][i]=[]
          autoVecPair[j][i].append((j,i))
          #autoVecPair[j][i].append((i,j))
          autoVecGraph[j][i].append(GPrev)
          #autoVecGraph[j][i].append(GPrevRev)
          
          first=False
        else:
          #for jtnd, jt in enumerate(list(G.nodes)[0:jnd]):
          #for jt in [i,j]:
            
            for knd,k in enumerate(autoVecGraph[j].keys()):
                if done ==False and is_automorphic_fast(autoVecGraph[j][k][0],GPrev):
                  
                  #print("graph where",j,"merged into",i,"already found in",j,"merged into ",k)
                  autoVecGraph[j][k].append(GPrev)
                  #autoVecGraph[j][k].append(GPrevRev)
                  autoVecPair[j][k].append((j,i))
                  #autoVecPair[j][k].append((i,j))
                  done=True
                #else:
                #  print("graph where",j,"merged into",i,"NOT ISO TO",jt,"merged into ",k)
            '''if done==False:
              for jtnd, jt in enumerate(list(G.nodes)[0:jnd]):
                if i in autoVecGraph[jt].keys():
                  if done ==False and (is_automorphic_fast(autoVecGraph[jt][i][0],GPrev)):
                      print("useful")
                      #print("graph where",j,"merged into",i,"already found in",j,"merged into ",k)
                      autoVecGraph[jt][k].append(GPrev)
                      autoVecPair[jt][k].append((j,i))
                      done=True'''
            if done==False:
              #if j==4:
                #print("adding",i,"to the auto vec of",j)
              autoVecPair[j][i]=[]
              autoVecGraph[j][i]=[]
              autoVecPair[j][i].append((j,i))
              #autoVecPair[j][i].append((i,j))
              autoVecGraph[j][i].append(GPrev)
              #autoVecGraph[j][i].append(GPrevRev)
  #print(autoVecPair[1])
  for jnd, j in enumerate(G.nodes):
    for knd,k in enumerate(autoVecPair[j].keys()):
      for lnd,l in enumerate(autoVecPair[j][k]):
        for mnd,m in enumerate(autoVecPair[j][k]):
          if l!=m:
            
            #print(autoVecPair[j][k][l[0]],autoVecPair[j][k][l][1],autoVecPair[j][k][m][0],autoVecPair[j][k][m][1])
            likelihood[autoVecPair[j][k][lnd][0]][autoVecPair[j][k][lnd][1]]=likelihood[autoVecPair[j][k][lnd][0]][autoVecPair[j][k][lnd][1]]+baseLikelihood[autoVecPair[j][k][mnd][0]][autoVecPair[j][k][mnd][1]]
            #likelihood[autoVecPair[j][k][mnd][0]][autoVecPair[j][k][mnd][1]]=likelihood[autoVecPair[j][k][mnd][0]][autoVecPair[j][k][mnd][1]]+baseLikelihood[autoVecPair[j][k][lnd][0]][autoVecPair[j][k][lnd][1]]
            #if j==1:
            #  if k==2 or k==3 or k==4 or k=='8_1':
            #    print(autoVecPair[j][k][lnd][0],autoVecPair[j][k][lnd][1],'cum',likelihood[autoVecPair[j][k][lnd][0]][autoVecPair[j][k][lnd][1]])
            #    print(autoVecPair[j][k][mnd][0],autoVecPair[j][k][mnd][1],'base',baseLikelihood[autoVecPair[j][k][mnd][0]][autoVecPair[j][k][mnd][1]])
  '''
      #likelihood[j][i]=baseLikelihood[j][i]
  for jnd, j in enumerate(G.nodes):
    for knd,k in enumerate(prevVec[j].keys()):
      for mnd, m in enumerate(G.nodes):   
        for lnd,l in enumerate(prevVec[m].keys()):  
          if jnd<=mnd:
            if knd<lnd:
              print("jnd",jnd,"knd",knd,"mnd",mnd,"lnd",lnd)      
              #print("aaaaaaaaaaaaaaaaaaaaaaaaaa")  
              if is_automorphic_fast(prevVec[j][k],prevVec[m][l]):
                #if j==5:
                #  print("knd",knd,"lnd",lnd)  
                print("graph where",j,"merged into",k,"already found")
            
                likelihood[j][k]=likelihood[j][k]+baseLikelihood[m][l]
                likelihood[m][l]=likelihood[m][l]+baseLikelihood[j][k]'''
  '''
  for jnd, j in enumerate(G.nodes):  
    for knd,k in enumerate(prevVec[j].keys()):
      for lnd,l in enumerate(prevVec[j].keys()):  
        if knd>lnd:      
            
          if is_automorphic_fast(prevVec[j][k],prevVec[j][l]):
            #if j==5:
            #  print("knd",knd,"lnd",lnd)  
            print("graph where",j,"merged into",k,"already found")
        
            likelihood[j][k]=likelihood[j][k]+baseLikelihood[j][l]
            likelihood[j][l]=likelihood[j][l]+baseLikelihood[j][k]'''
  return likelihood

def most_likely_previous_graph_no_auto(G,q,divide_by_n=True):
  prevVec=dict()
  autoVecPair=dict()
  autoVecGraph=dict()
  likelihood=dict()
  baseLikelihood=dict()
  first=True
  for j in G.nodes():
    prevVec[j]=dict()
    autoVecPair[j]=dict()
    autoVecGraph[j]=dict()
    likelihood[j]=dict()
    baseLikelihood[j]=dict()

  
  for jnd, j in enumerate(G.nodes):
    #first=True
    for ind,i in enumerate(list(G.nodes)):
    #for ind,i in enumerate(G.nodes):
      if i!=j:
        #print("dup",j,"anc",i)
        
        done=False
        GPrev=node_merger(G,i,j,self_loops=True)
        GPrevRev=node_merger(G,j,i,self_loops=True)
        prevVec[j][i]=GPrev
        
        
        if ind<jnd:
          baseLikelihood[j][i]=ancestral_likelihood_of_graph_PD(G,GPrev,i,j,q,divide_by_n)
          baseLikelihood[i][j]=baseLikelihood[j][i]
          likelihood[j][i]=baseLikelihood[j][i]
          likelihood[i][j]=baseLikelihood[i][j]
        '''
        if first==True:
          autoVecPair[j][i]=[]
          autoVecGraph[j][i]=[]
          autoVecPair[j][i].append((j,i))
          #autoVecPair[j][i].append((i,j))
          autoVecGraph[j][i].append(GPrev)
          #autoVecGraph[j][i].append(GPrevRev)
          
          first=False
        else:
          #for jtnd, jt in enumerate(list(G.nodes)[0:jnd]):
          #for jt in [i,j]:
            
            for knd,k in enumerate(autoVecGraph[j].keys()):
                if done ==False and is_automorphic_fast(autoVecGraph[j][k][0],GPrev):
                  
                  #print("graph where",j,"merged into",i,"already found in",j,"merged into ",k)
                  autoVecGraph[j][k].append(GPrev)
                  #autoVecGraph[j][k].append(GPrevRev)
                  autoVecPair[j][k].append((j,i))
                  #autoVecPair[j][k].append((i,j))
                  done=True
                #else:
                #  print("graph where",j,"merged into",i,"NOT ISO TO",jt,"merged into ",k)
          
            if done==False:
              #if j==4:
                #print("adding",i,"to the auto vec of",j)
              autoVecPair[j][i]=[]
              autoVecGraph[j][i]=[]
              autoVecPair[j][i].append((j,i))
              #autoVecPair[j][i].append((i,j))
              autoVecGraph[j][i].append(GPrev)
              #autoVecGraph[j][i].append(GPrevRev)
  for jnd, j in enumerate(G.nodes):
    for knd,k in enumerate(autoVecPair[j].keys()):
      for lnd,l in enumerate(autoVecPair[j][k]):
        for mnd,m in enumerate(autoVecPair[j][k]):
          if l!=m:
            #print(autoVecPair[j][k][l[0]],autoVecPair[j][k][l][1],autoVecPair[j][k][m][0],autoVecPair[j][k][m][1])
            likelihood[autoVecPair[j][k][lnd][0]][autoVecPair[j][k][lnd][1]]=likelihood[autoVecPair[j][k][lnd][0]][autoVecPair[j][k][lnd][1]]+baseLikelihood[autoVecPair[j][k][mnd][0]][autoVecPair[j][k][mnd][1]]
            #likelihood[autoVecPair[j][k][mnd][0]][autoVecPair[j][k][mnd][1]]=likelihood[autoVecPair[j][k][mnd][0]][autoVecPair[j][k][mnd][1]]+baseLikelihood[autoVecPair[j][k][lnd][0]][autoVecPair[j][k][lnd][1]]
'''
  return likelihood

def ancestral_likelihood_ped_pea(G,i,j,r,q,divide_by_n=True):
  G=copy.deepcopy(G)
  prod=1
  numNodes=len(list(G.nodes))
  anchorNeighboursIn = list(G.predecessors(i))
  anchorNeighboursOut = list(G.successors(i))
  duplicateNeighboursIn=list(G.predecessors(j))
  duplicateNeighboursOut=list(G.successors(j))
  intersectingNeighboursIn = list(set(anchorNeighboursIn)&set(duplicateNeighboursIn))
  intersectingNeighboursOut = list(set(anchorNeighboursOut)&set(duplicateNeighboursOut))
  intersectingNeighbours=intersectingNeighboursIn+intersectingNeighboursOut
  
  symmdiffNeighboursIn = list(set(anchorNeighboursIn)^set(duplicateNeighboursIn))
  symmdiffNeighboursOut = list(set(anchorNeighboursOut)^set(duplicateNeighboursOut))
  symmdiffNeighbours=symmdiffNeighboursIn+symmdiffNeighboursOut
  neighboursTot=len(intersectingNeighbours+symmdiffNeighbours)
  intersectNo=len(intersectingNeighbours)
  symmNo=len(symmdiffNeighbours)
  for k in intersectingNeighbours:
    prod = prod*(1-q)
  for k,l in enumerate(symmdiffNeighbours):
    qProd=1
    rProd=1
    rbProd=1
    for i in range(0,k):
      qProd=qProd*q
    for i in range(0,symmdiffNeighbours-k):
      rProd=rProd*(r/(numNodes-1))
    
    summer=summer+(qProd*rProd)
    #prod = prod*((theConst/neighboursTot)*q/2+((symmdiffNeighbours-theConst)/neighboursTot)*r/numNodes)
  prod=prod*summer
  if divide_by_n:
    prod=prod/numNodes
  
  return prod

def one_step_NK(G,qMod,qCon):
  G=copy.deepcopy(G)
  arrival=dict()
  anchor=dict()
  V=list(G.nodes)
  
  Vlen=len(V)
  G_anc=nx.DiGraph(G)
  
  maxLikelihood = -1
  pairList=[]
  for i in V:
    for j in V:
      if j!=i:
        tempLikelihood = ancestral_likelihood_dmc(G,i,j,qMod,qCon)
        if tempLikelihood==maxLikelihood:
          pairList.append((i,j))
        if tempLikelihood>maxLikelihood:
          pairList=[(i,j)]
          maxLikelihood=tempLikelihood
  if len(pairList)==0:
    return G,arrival,anchor
  rando =np.random.random()
  rando = int(np.round(rando*len(pairList)))
    
  selectedNodes=pairList[rando-1]
  ##print("Merge nodes",selectedNodes)
  arrival[selectedNodes[1]]=len(V)
  anchor[selectedNodes[1]]=selectedNodes[0]
  anchorNeighboursIn = list(G.predecessors(selectedNodes[0]))
  anchorNeighboursOut=list(G.successors(selectedNodes[0]))
  duplicateNeighboursIn=list(G.predecessors(selectedNodes[1]))
  duplicateNeighboursOut=list(G.successors(selectedNodes[1]))
  s = set(anchorNeighboursIn)
  anchorNeighboursIn = [x for x in duplicateNeighboursIn if x not in s]
  for i in anchorNeighboursIn:
    if i!=selectedNodes[1] and i in V:
      G_anc.add_edge(i,selectedNodes[0])
  s = set(anchorNeighboursOut)
  anchorNeighboursOut = [x for x in duplicateNeighboursOut if x not in s]
  for i in anchorNeighboursOut:
    if i!=selectedNodes[1] and i in V:
      G_anc.add_edge(selectedNodes[0],i)
  G_anc.remove_node(selectedNodes[1])
  V.remove(selectedNodes[1])
  Vlen=len(V)
  ##print("G_anc modified nodes",G_anc.nodes)
  return G_anc,arrival,anchor
def NK(G,k,qMod,qCon):
  G=copy.deepcopy(G)
  arrival=dict()
  anchor=dict()
  V=list(G.nodes)
  
  Vlen=len(V)
  G_anc=nx.DiGraph(G)
  ##print("G_anc original nodes",G_anc.nodes)
  while Vlen>k:
    maxLikelihood = -1
    pairList=[]
    for i in V:
      for j in V:
        if j!=i:
          tempLikelihood = ancestral_likelihood_dmc(G,i,j,qMod,qCon)
          
          if tempLikelihood==maxLikelihood:
            
            pairList.append((i,j))
            
            
          if tempLikelihood>maxLikelihood:
            pairList=[(i,j)]
            
            maxLikelihood=tempLikelihood
    if len(pairList)==0:
      return G,arrival,anchor
    rando =np.random.random()
    rando = int(np.round(rando*len(pairList)))
    
    selectedNodes=pairList[rando-1]
    ##print("Merge nodes",selectedNodes)
    arrival[selectedNodes[1]]=len(V)
    anchor[selectedNodes[1]]=selectedNodes[0]
    anchorNeighboursIn = list(G.predecessors(selectedNodes[0]))
    anchorNeighboursOut=list(G.successors(selectedNodes[0]))
    duplicateNeighboursIn=list(G.predecessors(selectedNodes[1]))
    duplicateNeighboursOut=list(G.successors(selectedNodes[1]))
    s = set(anchorNeighboursIn)
    anchorNeighboursIn = [x for x in duplicateNeighboursIn if x not in s]
    for i in anchorNeighboursIn:
      if i!=selectedNodes[1] and i in V:
        G_anc.add_edge(i,selectedNodes[0])
    s = set(anchorNeighboursOut)
    anchorNeighboursOut = [x for x in duplicateNeighboursOut if x not in s]
    for i in anchorNeighboursOut:
      if i!=selectedNodes[1] and i in V:
        G_anc.add_edge(selectedNodes[0],i)
    G_anc.remove_node(selectedNodes[1])
    V.remove(selectedNodes[1])
    Vlen=len(V)
    ##print("G_anc modified nodes",G_anc.nodes)
  return G_anc,arrival,anchor
def ancestral_pair(G1,G2,U1,U2,qMod,qCon):
  G1=copy.deepcopy(G1)
  G2=copy.deepcopy(G2)
  G1_induced=nx.induced_subgraph(G1,U1)
  G2_induced=nx.induced_subgraph(G2,U2)
  G1_reduced=G1_induced
  G2_reduced=G2_induced
  x=2
  t=0
  if len(U1)<len(U2):
    ##print("G2")
    t=len(G1_reduced.nodes)-len(G2_reduced.nodes)
    G2_reduced,arrival,anchor=NK(G2_induced,len(U1),qMod,qCon)
    
  if len(U1)>len(U2):
    ##print("G1")
    t=len(G2_reduced.nodes)-len(G1_reduced.nodes)
    G1_reduced,arrival,anchor=NK(G1_induced,len(U2),qMod,qCon)
    x=1
  
  return G1_reduced,G2_reduced,x,t
def gene_family_partitioner(G,orig_label=False):
  if orig_label:
    origin=""
    partition = dict()
    for i in list(G.nodes):
      if "_" in str(G.nodes[i]['orig_label']):
        origin=""
        s=str(G.nodes[i]['orig_label'])[0]
        j=0
        while s!='_':
          
          origin=origin+s
          j=j+1
          
          s=str(G.nodes[i]['orig_label'])[j]
          
      else:
        origin=str(G.nodes[i]['orig_label'])
      if origin not in partition:
        partition[origin]=[]
      partition[origin].append(i)
    
    return partition
  else:
    origin=""
    partition = dict()
    for i in list(G.nodes):
      if "_" in str(i):
        origin=""
        s=str(i)[0]
        j=0
        while s!='_':
          
          origin=origin+s
          j=j+1
          
          s=str(i)[j]
          
      else:
        origin=str(i)
      if origin not in partition:
        partition[origin]=[]
      partition[origin].append(i)
    
    return partition
def gene_family_relabeller(G):
  origin=""
  mapping = dict()
  originCounter=dict()
  leRand=copy.deepcopy(G.nodes())
  random.shuffle(list(G.nodes()))
  print(leRand)
  for i in leRand:
    if "_" in str(i):
      origin=""
      s=str(i)[0]
      j=0
      while s!='_':
        
        origin=origin+s
        j=j+1
        
        s=str(i)[j]
        
    else:
      origin=str(i)
    if origin not in originCounter:
      originCounter[origin]=0
    originCounter[origin]=originCounter[origin]+1
    mapping[i]=str(origin)+"_"+str(originCounter[origin])
  return mapping
def dmc_anc_rec(G1,G2,qMod,qCon):
  P=[]
  E0=[]
  P1 = gene_family_partitioner(G1)
  ##print("Partition G1",P1)
  P2 = gene_family_partitioner(G2)
  ##print("Partition G2",P2)
  if len(P1)<len(P2):
    bigP=P2
    smallP=P1
    Plen=len(smallP)
  else:
    bigP=P1
    smallP=P2
    Plen=len(smallP)
  for i in bigP:
    if i in smallP:
      
      P.append(ancestral_pair(G1,G2,P1[i],P2[i],qMod,qCon))
  B=sorted(P, key=lambda tup: tup[3])
  ##print(B,"B")
  V0=[]
  for i in range(0,len(B)):
    x=B[i][2]
    V0=V0+list((B[i][x-1]).nodes)
    ##print(B[i][x-1].nodes)
    for j in V0:
      if x==1:
        
        G1_induced=nx.induced_subgraph(G1,list(B[i][x-1].nodes))
        
        for u in list(B[i][x-1].nodes):
          if (u,j) in list(G1.edges):
            E0.append((u,j))
          if (j,u) in list(G1.edges):
            E0.append((j,u))
      if x==2:
        G2_induced=nx.induced_subgraph(G2,list(B[i][x-1].nodes))
        for u in list(B[i][x-1].nodes):
          if (u,j) in list(G2.edges):
            E0.append((u,j))
          if(j,u) in list(G2.edges):
            E0.append((j,u))
  
  gAncestral=nx.DiGraph()
  gAncestral.add_nodes_from(V0)
  gAncestral.add_edges_from(E0)
  return gAncestral
def pairs_will_give_automorphic_graph(G,i,j,k,l,self_loops=True):
  
  anchorNeighboursIn1 = list(G.predecessors(i))
  anchorNeighboursOut1=list(G.successors(i))
  duplicateNeighboursIn1=list(G.predecessors(j))
  duplicateNeighboursOut1=list(G.successors(j))

  s = set(anchorNeighboursIn1)
  anchorNeighboursIn1 = [(x,i) for x in duplicateNeighboursIn1 if x not in s and x!=j]

  s = set(anchorNeighboursOut1)
  anchorNeighboursOut1 = [(i,x) for x in duplicateNeighboursOut1 if x not in s and x!=j]
 
  anchorNeighboursIn2 = list(G.predecessors(k))
  anchorNeighboursOut2=list(G.successors(k))
  duplicateNeighboursIn2=list(G.predecessors(l))
  duplicateNeighboursOut2=list(G.successors(l))

  s = set(anchorNeighboursIn2)
  anchorNeighboursIn2 = [(x,k) for x in duplicateNeighboursIn2 if x not in s and x!=l]

  s = set(anchorNeighboursOut2)
  anchorNeighboursOut2 = [(k,x) for x in duplicateNeighboursOut2 if x not in s and x!=l]
     
  return len(set(anchorNeighboursIn1)^set(anchorNeighboursIn2))==0 and len(set(anchorNeighboursOut1)^set(anchorNeighboursOut2))==0
def node_merger(G,i,j,self_loops=True):
  
  #G_anc=copy.deepcopy(G)
  
  G_anc=nx.DiGraph()
  G_anc.add_nodes_from(G.nodes)
  G_anc.add_edges_from(G.edges)

  anchorNeighboursIn = list(G.predecessors(i))
  anchorNeighboursOut=list(G.successors(i))
  duplicateNeighboursIn=list(G.predecessors(j))
  duplicateNeighboursOut=list(G.successors(j))

  s = set(anchorNeighboursIn)
  anchorNeighboursIn = [(x,i) for x in duplicateNeighboursIn if x not in s and x!=j]
  G_anc.add_edges_from(anchorNeighboursIn)
  if not self_loops and i in anchorNeighboursIn:
          G_anc.remove_edge(i,i)
  '''for k in anchorNeighboursIn:
    if k!=j and k in list(G.nodes):
      if k==i:
        if self_loops:
          G_anc.add_edge(k,i)
      else:
        G_anc.add_edge(k,i)
  '''
  s = set(anchorNeighboursOut)
  anchorNeighboursOut = [(i,x) for x in duplicateNeighboursOut if x not in s and x!=j]
  G_anc.add_edges_from(anchorNeighboursOut)
  if not self_loops and i in anchorNeighboursIn:
          G_anc.remove_edge(i,i)
  '''for k in anchorNeighboursOut:
    if k!=j and k in list(G.nodes):
      if k==i:
        if self_loops:
          G_anc.add_edge(i,k)
      else:
        G_anc.add_edge(i,k)'''
  G_anc.remove_node(j)
  return G_anc


def GFAF_ped_pea(G1,G2,P1,P2,r,q,tolerance=0,toleranceEC=0,true_labels=True):
  """Returns a pair of networkx graphs G1' and G2' based off of the input graphs G1 and G2
    with nodes merged as to maximise their EC similarity.

    This is a heuristic algorithm that finds pairs of nodes, either in G1 and G2, that are the most likely to have been the most
    recently duplicated pairs and determines which of these pairs, when merged, will increase the
    EC similarity of the two graphs.

    The best pair under these conditions is chosen and is merged, and the process is repeated until the EC score 
    can no longer be improved. That is, a local optimum is found.

    Parameters
    ----------
    G1 : networkx graph
        One of the two graphs
    G2 : networkx graph
        One of the two graphs
    qMod : float
        The probability of edge deletion in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    qCon : float
        The probability of creating an edge between duplicated pairs of vertices in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    tolerance : float, optional (default=0.05)
        A tolerance value used to determine the acceptable likelihood a pair of nodes to have
        been the most recently duplicated pair. A non-zero tolerance allows less likely pairs to be chosen, possibly 
        increasing the EC score more than the best pair.

    Notes
    -----
    This algorithm is not guaranteed to find the global optimum. 

    This algorithm comprises 4 main steps:
    1. The likelihood of all pairs of nodes in both G1 and G2 are calculated.
    2. A list of the 'best' pairs is generated. These pairs are those that have the highest likelihood, 
    depending on the likelihood tolerance value.
    3. From these pairs, the pair(s) that improve(s) the EC score the most is chosen and merged. Most of the time there
    is more than one such pair, in which case a random one is chosen.
    4. The graph in which the most recent merge has occured is updated and the process is repeated until the EC score 
    can no longer be improved.
    
    Issues 
    ------
    1. For the first go of the loop, the maxScore is -1 and so all the scores are greater than it
    2. It is assumed that likelihoods are comparable across the graphs G1 and G2 for different sized graphs. 
    This very well may not be the case and should be followed up.
    3. Many pairs seem to have the give the same EC score. This could be fine, but possibly something is going on here.

    """
  
  if true_labels:
    #graphPair is a repeatedly updated tuple containing the present state of graphs G1 and G2, the EC score of the current pair,
    # and an int that represents which of G1 and G2 the most recent merge occured in.
    graphPair=(G1,G2,-1,0) 
    #signal is a boolean that is set to False when the EC score of the current pair is no longer improving.
    signal =True 
    
    
    #maxScore records the current best EC score
    maxScore=-1
    n=len(G1.nodes())
    theGraphList=[]
    
    prevScore=conserved_edges(G1,G2)
    countUp=False
    while signal and len(graphPair[0].nodes)>4:
      #step 1: calculate the likelihood of all pairs of nodes in both G1 and G2

      #alignedPairs is a dictionary, keyed by (i,j,k) that records the EC score of graphs resulting from merging 
      # each pair of nodes i,j in graph G_k (k=1 or k=2).
      alignedPairs=[]
      #maxLikelihood is updated each loop and records the likelihood score of the most likely node pair for the current loop
      maxLikelihood = -1
      #pairList is a list of tuples (i,j,k,likelihood) where i and j are the nodes in graph G_k
      # and likelihood is the likelihood of the pair (i,j) being most recently duplicated in graph G_k
      pairList=[]
      #G1 and G2 are updated to be the most recent pair of graphs
      G1=graphPair[0]
      G2=graphPair[1]
      P1=gene_family_partitioner(G1,orig_label=True)
      P2=gene_family_partitioner(G2,orig_label=True)
      print("p1,p2",P1,P2)
      external_nodes_1=dict()
      external_nodes_2=dict()
      for fam in P1:
        external_nodes_1[fam]=[i for i in P1[fam] if i not in P2[fam]]
      print("external",external_nodes_1)
      #Consider each node pair (i,j) in G1

      for fam in P1:
        for num1,i in enumerate(external_nodes_1[fam]):
          for num2,j in enumerate(P1[fam]):
            
            if i!=j:
              #Construct the graph G1_temp resulting from merging (i,j) in G1
              #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
              
              #check the likelihood function for the pair (i,j)
              tempLikelihood = ancestral_likelihood_ped_pea(G1,i,j,r,q)
              
              #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
              # of the merged node graph G1_temp and G2
              pairList.append((i,j,1,tempLikelihood))
              
      
      #Consider each node pair (i,j) in G2
      for fam in P2:
        external_nodes_2[fam]=[i for i in P2[fam] if i not in P1[fam]]
      for fam in P2:
        for num1,i in enumerate(external_nodes_2[fam]):  
          for num2,j in enumerate(P2[fam]):
            if i!=j:
              #Construct the graph G2_temp resulting from merging (i,j) in G2
              #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
              
              
              #check the likelihood function for the pair (i,j)
              tempLikelihood = ancestral_likelihood_ped_pea(G2,i,j,r,q)
              
              #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
              # of the merged node graph G2_temp and G1
              pairList.append((i,j,2,tempLikelihood))
              
              
      #step 2: choose the best pair(s) based on the likelihood function and the tolerance value
      #End algorithm if pairList is empty. Only occurs if both graphs are single node or empty.
      if len(pairList)==0:
        break
      maxLikelihood=max(pairList,key=lambda x:x[3])[3]
      
      pairPairList=[]      
      #step 2: choose the best pair(s) based on the likelihood function and the tolerance value
      #End algorithm if pairList is empty. Only occurs if both graphs are single node or empty.
      if pairList==[]:
        signal=False
      #if pairList isn't empty, choose the best pair(s) based on the likelihood function and the tolerance value  
      else:
        print("list begins")
        for i in pairList:
          if i[3]>= maxLikelihood-tolerance*maxLikelihood:
                pairPairList.append((i[0],i[1],i[2],i[3]))
        
        print("sorting begins")
        pairPairList=sorted(pairPairList, key=lambda x: x[3],reverse=True) #find the maximum likelihood over all pairs
        print("merging begins")
        count=0
        signal=False
        
        trigger=False
        chosenLikelihood=0
        countUp=False
        bestScoreVec=[]
        for i in pairPairList:
          count=count+1
          if count>=len(pairPairList):
            countUp=True
            break
          if trigger and i[3]<chosenLikelihood:
            print('ay you broke my trigger')
            break
          if i[2]==1:
            if i[1] not in G2.nodes and i[0] in G2.nodes:
              G1_temp=node_merger(G1,i[0],i[1],self_loops=True)
            else:
              G1_temp=node_merger(G1,i[1],i[0],self_loops=True)
            G1_tempp=copy.deepcopy(G1_temp)
            tempScore= conserved_edges(G1_tempp,G2)
            print(i,tempScore)
            if tempScore>=prevScore:
              if tempScore>prevScore:
                bestScoreVec=[]
              print("chosen",i)
              graphPair=(G1_temp,G2,tempScore)
              prevScore=tempScore
              trigger=True
              chosenLikelihood=i[3]
              signal=True
              bestScoreVec.append(graphPair)
              
          if i[2]==2:
            if i[1] not in G1.nodes and i[0] in G1.nodes:
              G2_temp=node_merger(G2,i[0],i[1],self_loops=True)
            else:
              G2_temp=node_merger(G2,i[1],i[0],self_loops=True)
            G2_tempp=copy.deepcopy(G2_temp)
            tempScore= conserved_edges(G1,G2_tempp)
            print(i,tempScore)
            if tempScore>=prevScore:
              if tempScore>prevScore:
                bestScoreVec=[]
              print("chosen",i)
              graphPair=(G1,G2_temp,tempScore)
              prevScore=tempScore
              trigger=True
              chosenLikelihood=i[3]
              signal=True
              bestScoreVec.append(graphPair)
        
          
        print("out of the merge")
        if len(bestScoreVec)!=0:
          graphPair=random.choice(bestScoreVec)
        theGraphList.append(graphPair)
        print(graphPair,len(graphPair[0].nodes))  
    
      
    #graphPair=max(theNewGraphList,key=lambda x:x[2])
    print("the big winner score is",graphPair[2],"and its graphs with",len(graphPair[0].nodes),"nodes")
    return graphPair[0],graphPair[1] 
  else:
    #graphPair is a repeatedly updated tuple containing the present state of graphs G1 and G2, the EC score of the current pair,
    # and an int that represents which of G1 and G2 the most recent merge occured in.
    graphPair=(G1,G2,-1,0) 
    #signal is a boolean that is set to False when the EC score of the current pair is no longer improving.
    signal =True 
    
    
    #maxScore records the current best EC score
    maxScore=-1
    n=len(G1.nodes())
    theGraphList=[]
    P1=gene_family_partitioner(G1,orig_label=True)
    P2=gene_family_partitioner(G2,orig_label=True)
    prevScore=conserved_edges(G1,G2)
    countUp=False
    while signal and len(graphPair[0].nodes)>4:
      #step 1: calculate the likelihood of all pairs of nodes in both G1 and G2

      #alignedPairs is a dictionary, keyed by (i,j,k) that records the EC score of graphs resulting from merging 
      # each pair of nodes i,j in graph G_k (k=1 or k=2).
      alignedPairs=[]
      #maxLikelihood is updated each loop and records the likelihood score of the most likely node pair for the current loop
      maxLikelihood = -1
      #pairList is a list of tuples (i,j,k,likelihood) where i and j are the nodes in graph G_k
      # and likelihood is the likelihood of the pair (i,j) being most recently duplicated in graph G_k
      pairList=[]
      #G1 and G2 are updated to be the most recent pair of graphs
      G1=graphPair[0]
      G2=graphPair[1]
      P1=gene_family_partitioner(G1)
      P2=gene_family_partitioner(G2)
      
      
      #Consider each node pair (i,j) in G1

      for fam in P1:
        for num1,i in enumerate(P1[fam]):
          for num2,j in enumerate(P1[fam]):
            
            if num2>num1:
              #Construct the graph G1_temp resulting from merging (i,j) in G1
              #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
              
              #check the likelihood function for the pair (i,j)
              tempLikelihood = ancestral_likelihood_ped_pea(G1,i,j,r,q)
              
              #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
              # of the merged node graph G1_temp and G2
              pairList.append((i,j,1,tempLikelihood))
              
      
      #Consider each node pair (i,j) in G2
      
      for fam in P2:
        for num1,i in enumerate(P2[fam]):  
          for num2,j in enumerate(P2[fam]):
            if num2>num1:
              #Construct the graph G2_temp resulting from merging (i,j) in G2
              #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
              
              
              #check the likelihood function for the pair (i,j)
              tempLikelihood = ancestral_likelihood_ped_pea(G2,i,j,r,q)
              
              #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
              # of the merged node graph G2_temp and G1
              pairList.append((i,j,2,tempLikelihood))
              
              
      #step 2: choose the best pair(s) based on the likelihood function and the tolerance value
      #End algorithm if pairList is empty. Only occurs if both graphs are single node or empty.
      if len(pairList)==0:
        break
      maxLikelihood=max(pairList,key=lambda x:x[3])[3]
      
      pairPairList=[]      
      #step 2: choose the best pair(s) based on the likelihood function and the tolerance value
      #End algorithm if pairList is empty. Only occurs if both graphs are single node or empty.
      if pairList==[]:
        signal=False
      #if pairList isn't empty, choose the best pair(s) based on the likelihood function and the tolerance value  
      else:
        print("list begins")
        for i in pairList:
          if i[3]>= maxLikelihood-tolerance*maxLikelihood:
                pairPairList.append((i[0],i[1],i[2],i[3]))
        
        print("sorting begins")
        pairPairList=sorted(pairPairList, key=lambda x: x[3],reverse=True) #find the maximum likelihood over all pairs
        print("merging begins")
        count=0
        signal=False
        
        trigger=False
        chosenLikelihood=0
        countUp=False
        bestScoreVec=[]
        for i in pairPairList:
          count=count+1
          if count>len(pairPairList):
            countUp=True
            break
          if trigger and i[3]<chosenLikelihood:
            print('ay you broke my trigger')
            break
          if i[2]==1:
            if i[1] not in G2.nodes and i[0] in G2.nodes:
              G1_temp=node_merger(G1,i[0],i[1],self_loops=True)
            else:
              G1_temp=node_merger(G1,i[1],i[0],self_loops=True)
            G1_tempp=copy.deepcopy(G1_temp)
            tempScore= conserved_edges(G1_tempp,G2)
            print(i,tempScore)
            if tempScore>=prevScore:
              if tempScore>prevScore:
                bestScoreVec=[]
              print("chosen",i)
              graphPair=(G1_temp,G2,tempScore)
              prevScore=tempScore
              trigger=True
              chosenLikelihood=i[3]
              signal=True
              bestScoreVec.append(graphPair)
              
          if i[2]==2:
            if i[1] not in G1.nodes and i[0] in G1.nodes:
              G2_temp=node_merger(G2,i[0],i[1],self_loops=True)
            else:
              G2_temp=node_merger(G2,i[1],i[0],self_loops=True)
            G2_tempp=copy.deepcopy(G2_temp)
            tempScore= conserved_edges(G1,G2_tempp)
            print(i,tempScore)
            if tempScore>=prevScore:
              if tempScore>prevScore:
                bestScoreVec=[]
              print("chosen",i)
              graphPair=(G1,G2_temp,tempScore)
              prevScore=tempScore
              trigger=True
              chosenLikelihood=i[3]
              signal=True
              bestScoreVec.append(graphPair)
        
          
        print("out of the merge")
        if len(bestScoreVec)!=0:
          graphPair=random.choice(bestScoreVec)
        theGraphList.append(graphPair)
        print(graphPair,len(graphPair[0].nodes))  
    
      
    #graphPair=max(theNewGraphList,key=lambda x:x[2])
    print("the big winner score is",graphPair[2],"and its graphs with",len(graphPair[0].nodes),"nodes")
    return graphPair[0],graphPair[1]        
      
def GFAF_dmc(G1,G2,P1,P2,qCon,qMod,tolerance=0,toleranceEC=0,true_labels=True):


    
  """Returns a pair of networkx graphs G1' and G2' based off of the input graphs G1 and G2
    with nodes merged as to maximise their EC similarity.

    This is a heuristic algorithm that finds pairs of nodes, either in G1 and G2, that are the most likely to have been the most
    recently duplicated pairs and determines which of these pairs, when merged, will increase the
    EC similarity of the two graphs.

    The best pair under these conditions is chosen and is merged, and the process is repeated until the EC score 
    can no longer be improved. That is, a local optimum is found.

    Parameters
    ----------
    G1 : networkx graph
        One of the two graphs
    G2 : networkx graph
        One of the two graphs
    qMod : float
        The probability of edge deletion in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    qCon : float
        The probability of creating an edge between duplicated pairs of vertices in the assumed forward DMC model.
        In cases where this is not known it may need to be estimated.
    tolerance : float, optional (default=0.05)
        A tolerance value used to determine the acceptable likelihood a pair of nodes to have
        been the most recently duplicated pair. A non-zero tolerance allows less likely pairs to be chosen, possibly 
        increasing the EC score more than the best pair.

    Notes
    -----
    This algorithm is not guaranteed to find the global optimum. 

    This algorithm comprises 4 main steps:
    1. The likelihood of all pairs of nodes in both G1 and G2 are calculated.
    2. A list of the 'best' pairs is generated. These pairs are those that have the highest likelihood, 
    depending on the likelihood tolerance value.
    3. From these pairs, the pair(s) that improve(s) the EC score the most is chosen and merged. Most of the time there
    is more than one such pair, in which case a random one is chosen.
    4. The graph in which the most recent merge has occured is updated and the process is repeated until the EC score 
    can no longer be improved.
    
    Issues 
    ------
    1. For the first go of the loop, the maxScore is -1 and so all the scores are greater than it
    2. It is assumed that likelihoods are comparable across the graphs G1 and G2 for different sized graphs. 
    This very well may not be the case and should be followed up.
    3. Many pairs seem to have the give the same EC score. This could be fine, but possibly something is going on here.

    """
  
  if true_labels:
    #graphPair is a repeatedly updated tuple containing the present state of graphs G1 and G2, the EC score of the current pair,
    # and an int that represents which of G1 and G2 the most recent merge occured in.
    graphPair=(G1,G2,-1,0) 
    #signal is a boolean that is set to False when the EC score of the current pair is no longer improving.
    signal =True 
    
    
    #maxScore records the current best EC score
    maxScore=-1
    n=len(G1.nodes())
    theGraphList=[]
    
    prevScore=conserved_edges(G1,G2)
    countUp=False
    while signal and len(graphPair[0].nodes)>4:
      #step 1: calculate the likelihood of all pairs of nodes in both G1 and G2

      #alignedPairs is a dictionary, keyed by (i,j,k) that records the EC score of graphs resulting from merging 
      # each pair of nodes i,j in graph G_k (k=1 or k=2).
      alignedPairs=[]
      #maxLikelihood is updated each loop and records the likelihood score of the most likely node pair for the current loop
      maxLikelihood = -1
      #pairList is a list of tuples (i,j,k,likelihood) where i and j are the nodes in graph G_k
      # and likelihood is the likelihood of the pair (i,j) being most recently duplicated in graph G_k
      pairList=[]
      #G1 and G2 are updated to be the most recent pair of graphs
      G1=graphPair[0]
      G2=graphPair[1]
      P1=gene_family_partitioner(G1,orig_label=True)
      P2=gene_family_partitioner(G2,orig_label=True)
      print("p1,p2",P1,P2)
      external_nodes_1=dict()
      external_nodes_2=dict()
      for fam in P1:
        external_nodes_1[fam]=[i for i in P1[fam] if i not in P2[fam]]
      print("external",external_nodes_1)
      #Consider each node pair (i,j) in G1

      for fam in P1:
        for num1,i in enumerate(external_nodes_1[fam]):
          for num2,j in enumerate(P1[fam]):
            
            if i!=j:
              #Construct the graph G1_temp resulting from merging (i,j) in G1
              #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
              
              #check the likelihood function for the pair (i,j)
              tempLikelihood = ancestral_likelihood_dmc(G1,i,j,qMod,qCon)
              
              #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
              # of the merged node graph G1_temp and G2
              pairList.append((i,j,1,tempLikelihood))
              
      
      #Consider each node pair (i,j) in G2
      for fam in P2:
        external_nodes_2[fam]=[i for i in P2[fam] if i not in P1[fam]]
      for fam in P2:
        for num1,i in enumerate(external_nodes_2[fam]):  
          for num2,j in enumerate(P2[fam]):
            if i!=j:
              #Construct the graph G2_temp resulting from merging (i,j) in G2
              #This pair of if statements ensures that a node merge keeps a label common to both graphs if possible
              
              
              #check the likelihood function for the pair (i,j)
              tempLikelihood = ancestral_likelihood_dmc(G2,i,j,qMod,qCon)
              
              #Update the pairList and the alignedPairs dictionary with the likelihood of the pair (i,j) and the EC score
              # of the merged node graph G2_temp and G1
              pairList.append((i,j,2,tempLikelihood))
              
              
      #step 2: choose the best pair(s) based on the likelihood function and the tolerance value
      #End algorithm if pairList is empty. Only occurs if both graphs are single node or empty.
      if len(pairList)==0:
        break
      maxLikelihood=max(pairList,key=lambda x:x[3])[3]
      
      pairPairList=[]      
      #step 2: choose the best pair(s) based on the likelihood function and the tolerance value
      #End algorithm if pairList is empty. Only occurs if both graphs are single node or empty.
      if pairList==[]:
        signal=False
      #if pairList isn't empty, choose the best pair(s) based on the likelihood function and the tolerance value  
      else:
        print("list begins")
        for i in pairList:
          if i[3]>= maxLikelihood-tolerance*maxLikelihood:
                pairPairList.append((i[0],i[1],i[2],i[3]))
        
        print("sorting begins")
        pairPairList=sorted(pairPairList, key=lambda x: x[3],reverse=True) #find the maximum likelihood over all pairs
        print("merging begins")
        count=0
        signal=False
        
        trigger=False
        chosenLikelihood=0
        countUp=False
        bestScoreVec=[]
        for i in pairPairList:
          count=count+1
          if count>=len(pairPairList):
            countUp=True
            break
          if trigger and i[3]<chosenLikelihood:
            print('ay you broke my trigger')
            break
          if i[2]==1:
            if i[1] not in G2.nodes and i[0] in G2.nodes:
              G1_temp=node_merger(G1,i[0],i[1],self_loops=True)
            else:
              G1_temp=node_merger(G1,i[1],i[0],self_loops=True)
            G1_tempp=copy.deepcopy(G1_temp)
            tempScore= conserved_edges(G1_tempp,G2)
            print(i,tempScore)
            if tempScore>=prevScore:
              if tempScore>prevScore:
                bestScoreVec=[]
              print("chosen",i)
              graphPair=(G1_temp,G2,tempScore)
              prevScore=tempScore
              trigger=True
              chosenLikelihood=i[3]
              signal=True
              bestScoreVec.append(graphPair)
              
          if i[2]==2:
            if i[1] not in G1.nodes and i[0] in G1.nodes:
              G2_temp=node_merger(G2,i[0],i[1],self_loops=True)
            else:
              G2_temp=node_merger(G2,i[1],i[0],self_loops=True)
            G2_tempp=copy.deepcopy(G2_temp)
            tempScore= conserved_edges(G1,G2_tempp)
            print(i,tempScore)
            if tempScore>=prevScore:
              if tempScore>prevScore:
                bestScoreVec=[]
              print("chosen",i)
              graphPair=(G1,G2_temp,tempScore)
              prevScore=tempScore
              trigger=True
              chosenLikelihood=i[3]
              signal=True
              bestScoreVec.append(graphPair)
        
          
        print("out of the merge")
        if len(bestScoreVec)!=0:
          graphPair=random.choice(bestScoreVec)
        theGraphList.append(graphPair)
        print(graphPair,len(graphPair[0].nodes))  
    
    print("the big winner score is",graphPair[2],"and its graphs with",len(graphPair[0].nodes),"nodes")
    return graphPair[0],graphPair[1]     


def two_cherry_tree_ancestral(model="ped_pea",algType="gene",branchLength=1,qMod=0.4,qCon=0.1,r=0.1,q=0.4,direc="test_datasets_ancestral/regulatory_test_ped_pea/",zhu_nakleh=True,noalg=True,noalg_align=True,myalg=True,myalg_align=True):

  leafGraphs=dict()
  internalGraphs=dict()
  t=nx.DiGraph()
  t.add_edge(0,1)
  t.add_edge(1,2)
  t.add_edge(1,3)
  t.add_edge(0,4)
  t.add_edge(4,5)
  t.add_edge(4,6)
  root = [n for n,d in t.in_degree() if d==0]
  leaves = [n for n,d in t.out_degree() if d==0]
  for i in range(7):
      if i in leaves:
          leafGraphs[i]= nx.read_edgelist(direc+"anc50_2cherry_branch"+str(branchLength)+"/LEAF"+str(i)+"_two_cherry_branchLength"+str(branchLength)+".txt",create_using=nx.DiGraph)
          #leafGraphs[i]= nx.read_edgelist("test_datasets_ancestral/code_quickrun/LEAF"+str(i)+"_two_cherry_branchLength"+str(branchLength)+".txt",create_using=nx.DiGraph)

      else:
          #internalGraphs[i]= nx.read_edgelist("test_datasets_ancestral/code_quickrun/INTERNAL"+str(i)+"_two_cherry_branchLength"+str(branchLength)+".txt",create_using=nx.DiGraph)

          internalGraphs[i]= nx.read_edgelist(direc+"anc50_2cherry_branch"+str(branchLength)+"/INTERNAL"+str(i)+"_two_cherry_branchLength"+str(branchLength)+".txt",create_using=nx.DiGraph)
  G_anc_dict=dict()
  G_anc_dict[(2,3)]=internalGraphs[1]
  G_anc_dict[(3,2)]=internalGraphs[1]
  G_anc_dict[(5,6)]=internalGraphs[4]
  G_anc_dict[(6,5)]=internalGraphs[4]

  G_anc_dict[(2,5)]=internalGraphs[0]
  G_anc_dict[(5,2)]=internalGraphs[0]
  G_anc_dict[(6,2)]=internalGraphs[0]
  G_anc_dict[(2,6)]=internalGraphs[0]
  G_anc_dict[(3,6)]=internalGraphs[0]
  G_anc_dict[(6,3)]=internalGraphs[0]
  G_anc_dict[(3,5)]=internalGraphs[0]
  G_anc_dict[(5,3)]=internalGraphs[0]

  iterations=1

  iterationVec=[i for i in range(iterations)]

  S3_nf_vec=dict()
  S3_dmc_vec=dict()
  S3_orig_vec=dict()
  EC_nf_vec=dict()
  EC_dmc_vec=dict()
  EC_orig_vec=dict()
  ICS_nf_vec=dict()
  ICS_dmc_vec=dict()
  ICS_orig_vec=dict()
  new_metric_nf_vec=dict()
  new_metric_dmc_vec=dict()
  new_metric_orig_vec=dict()
  network_order_dmc=dict()
  network_order_nf=dict()
  network_order_orig=dict()
  EC_find1_vec=dict()
  ICS_find1_vec=dict()
  S3_find1_vec=dict()
  network_order_find1=dict()
  EC_find2_vec=dict()
  ICS_find2_vec=dict()
  S3_find2_vec=dict()
  network_order_find2=dict()
  EC_find_int_vec=dict()
  ICS_find_int_vec=dict()
  S3_find_int_vec=dict()
  network_order_find_int=dict()
  EC_find_mapped_int_vec=dict()
  ICS_find_mapped_int_vec=dict()
  S3_find_mapped_int_vec=dict()
  network_order_int_mapped=dict()
  EC_findAl1_vec=dict()
  ICS_findAl1_vec=dict()
  S3_findAl1_vec=dict()
  network_order_findAl1=dict()
  EC_findAl2_vec=dict()
  ICS_findAl2_vec=dict()
  S3_findAl2_vec=dict()
  network_order_findAl2=dict()
  EC_findAl_int_vec=dict()
  ICS_findAl_int_vec=dict()
  S3_findAl_int_vec=dict()
  network_order_findAl_int=dict()
  EC_findAl_mapped_int_vec=dict()
  ICS_findAl_mapped_int_vec=dict()
  S3_findAl_mapped_int_vec=dict()
  network_order_int_mapped=dict()
  EC_true_unint_vec=dict()
  ICS_true_unint_vec=dict()
  S3_true_unint_vec=dict()
  network_order_true_unint=dict()
  EC_align_unint_vec=dict()
  ICS_align_unint_vec=dict()
  S3_align_unint_vec=dict()
  network_order_align_unint=dict()
  EC_unint_vec=dict()
  ICS_unint_vec=dict()
  S3_unint_vec=dict()
  network_order_unint=dict()

  conserved_nf_vec=dict()
  conserved_dmc_vec=dict()
  conserved_orig_vec=dict()
  conserved_unint_vec=dict()
  conserved_true_unint_vec=dict()
  conserved_find_int_vec=dict()

  extra_edges_nf_vec=dict()
  extra_edges_dmc_vec=dict()
  extra_edges_orig_vec=dict()
  extra_edges_unint_vec=dict()
  extra_edges_true_unint_vec=dict()
  extra_edges_find_int_vec=dict()

  missed_edges_nf_vec=dict()
  missed_edges_dmc_vec=dict()
  missed_edges_orig_vec=dict()
  missed_edges_unint_vec=dict()
  missed_edges_true_unint_vec=dict()
  missed_edges_find_int_vec=dict()

  conserved_align_int_vec=dict()
  extra_edges_align_int_vec=dict()
  missed_edges_align_int_vec=dict()
  conserved_align_unint_vec=dict()
  extra_edges_align_unint_vec=dict()
  missed_edges_align_unint_vec=dict()

  conserved_nf_unint_vec=dict()
  extra_edges_nf_unint_vec=dict()
  missed_edges_nf_unint_vec=dict()
  EC_nf_unint_vec=dict()
  ICS_nf_unint_vec=dict()
  S3_nf_unint_vec=dict()
  network_order_nf_unint=dict()
  print(leafGraphs)
  for l in leafGraphs:
      for m in leafGraphs:
          if m>l:
              print(l,m)
              G1=leafGraphs[l]
              G2=leafGraphs[m]
              G_anc=G_anc_dict[(l,m)] 
              G_anc=label_conserver(G_anc)
              print("g_anc true edge number",len(G_anc.edges()))  
              G1=label_conserver(G1)
              G2=label_conserver(G2)
              
              for k in range(0,iterations):
                  print("qMod:",qMod,k)
                  if myalg:
                    #True label My ancestral algorithm

                    G1_orig=copy.deepcopy(G1)
                    G2_orig=copy.deepcopy(G2)
                    #mapper1=gene_family_relabeller(G1_orig)
                    #mapper2=gene_family_relabeller(G2_orig)
                    #G1_orig=nx.relabel_nodes(G1_orig,mapper1)
                    #G2_orig=nx.relabel_nodes(G2_orig,mapper2)
                    #initial alignment
                    '''
                    G1_labelless=nx.convert_node_labels_to_integers(G1_orig)
                    G2_labelless=nx.convert_node_labels_to_integers(G2_orig)
                    G1=nx.convert_node_labels_to_integers(G1_orig)
                    G2=nx.convert_node_labels_to_integers(G2_orig)
                    alignVec,mapped=NF(G1,G2,32,0.8)
                    mapping = dict(alignVec)
                    
                    G1_mapped=nx.induced_subgraph(G1,list(mapped))
                    G1_mapped=nx.relabel_nodes(G1_mapped,mapping)
                    '''

                    t=1
                    tEC=0
                    
                    if model=="ped_pea":
                      if algType=="gene":
                          G_anc_find1,G_anc_find2=GFAF_ped_pea(G1_orig,G2_orig,r,q,tolerance=t,toleranceEC=tEC,true_labels=True)
                    elif model=="dmc":
                      if algType=="gene":
                          G_anc_find1,G_anc_find2=GFAF_dmc(G1_orig,G2_orig,qMod,qCon,tolerance=t,toleranceEC=tEC)
                   
                    else:
                        exit
                    print("ancestor found")
                    G_find_int_anc=nx.intersection(G_anc_find1,G_anc_find2)
                    print("my alg edge number int",len(G_find_int_anc.edges()))
                    G_true_unint=graph_intersection_union(G_anc_find1,G_anc_find2)
                    print("my alg edge number unint",len(G_true_unint.edges()))
                    mapping=dict()
                    for i in list(G_find_int_anc.nodes):
                        mapping[i]=G_anc_find1.nodes[i]['orig_label']

                    G_true_unint=nx.relabel_nodes(G_true_unint,mapping)    
                    EC_true_unint_vec[str((l,m))]=normalised_ec_score(G_true_unint,G_anc)
                    ICS_true_unint_vec[str((l,m))]=ics_score(G_true_unint,G_anc)
                    S3_true_unint_vec[str((l,m))]=s3_score(G_true_unint,G_anc)
                    conservedEdges=conserved_edges(G_true_unint,G_anc)
                    conserved_true_unint_vec[str((l,m))]=conservedEdges
                    extra_edges_true_unint_vec[str((l,m))]=len(G_true_unint.edges)-conservedEdges
                    missed_edges_true_unint_vec[str((l,m))]=len(G_anc.edges)-conservedEdges
                    network_order_true_unint[str((l,m))]=len(G_true_unint.nodes)
                    for j in list(G_find_int_anc.nodes()):
                        if (G_find_int_anc.out_degree(j)==0 and G_find_int_anc.in_degree(j)==0):
                            G_find_int_anc.remove_node(j)
                        elif G_find_int_anc.out_degree(j)==1 and G_find_int_anc.in_degree(j)==1 and (j,j) in list(G_find_int_anc.edges):
                            G_find_int_anc.remove_node(j)
                    
                    
                    
                    G_find_int_anc=nx.relabel_nodes(G_find_int_anc,mapping)

                    EC_find_int_vec[str((l,m))]=normalised_ec_score(G_find_int_anc,G_anc)
                    ICS_find_int_vec[str((l,m))]=ics_score(G_find_int_anc,G_anc)
                    S3_find_int_vec[str((l,m))]=s3_score(G_find_int_anc,G_anc)
                    conservedEdges=conserved_edges(G_find_int_anc,G_anc)
                    conserved_find_int_vec[str((l,m))]=conservedEdges
                    extra_edges_find_int_vec[str((l,m))]=len(G_find_int_anc.edges)-conservedEdges
                    missed_edges_find_int_vec[str((l,m))]=len(G_anc.edges)-conservedEdges
                    network_order_find_int[str((l,m))]=len(G_find_int_anc.nodes)

                    for j in list(G_anc_find1.nodes()):
                        if (G_anc_find1.out_degree(j)==0 and G_anc_find1.in_degree(j)==0):
                            G_anc_find1.remove_node(j)
                        elif G_anc_find1.out_degree(j)==1 and G_anc_find1.in_degree(j)==1 and (j,j) in list(G_anc_find1.edges):
                            G_anc_find1.remove_node(j)

                    mapping=dict()
                    for i in list(G_anc_find1.nodes):
                        mapping[i]=G_anc_find1.nodes[i]['orig_label']
                    G_find1_anc=nx.relabel_nodes(G_anc_find1,mapping)
                    
                    EC_find1_vec[str((l,m))]=normalised_ec_score(G_find1_anc,G_anc)
                    ICS_find1_vec[str((l,m))]=ics_score(G_find1_anc,G_anc)
                    S3_find1_vec[str((l,m))]=s3_score(G_find1_anc,G_anc)
                    network_order_find1[str((l,m))]=len(G_find1_anc.nodes)
                    
                    for j in list(G_anc_find2.nodes()):
                        if (G_anc_find2.out_degree(j)==0 and G_anc_find2.in_degree(j)==0):
                            G_anc_find2.remove_node(j)
                        elif G_anc_find2.out_degree(j)==1 and G_anc_find2.in_degree(j)==1 and (j,j) in list(G_anc_find2.edges):
                            G_anc_find2.remove_node(j)
                    
                    mapping=dict()
                    for i in list(G_anc_find2.nodes):
                        mapping[i]=G_anc_find2.nodes[i]['orig_label']
                    G_find2_anc=nx.relabel_nodes(G_anc_find2,mapping)

                    EC_find2_vec[str((l,m))]=normalised_ec_score(G_find2_anc,G_anc)
                    ICS_find2_vec[str((l,m))]=ics_score(G_find2_anc,G_anc)
                    S3_find2_vec[str((l,m))]=s3_score(G_find2_anc,G_anc)
                    network_order_find2[str((l,m))]=len(G_find2_anc.nodes)

                    nx.write_edgelist(G_find_int_anc,direc+"anc50_2cherry_branch"+str(branchLength)+"/myalg_int_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                    nx.write_edgelist(G_true_unint,direc+"anc50_2cherry_branch"+str(branchLength)+"/myalg_unint_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                    nx.write_edgelist(G_find1_anc,direc+"anc50_2cherry_branch"+str(branchLength)+"/myalgG"+str(l)+"_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                    nx.write_edgelist(G_find2_anc,direc+"anc50_2cherry_branch"+str(branchLength)+"/myalgG"+str(m)+"_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                  if zhu_nakleh:  
                    #Zhu-Nakleh Reconstruction
                    G1_orig=copy.deepcopy(G1)
                    G2_orig=copy.deepcopy(G2)
                    G_dmc_anc=dmc_anc_rec(G1_orig,G2_orig,qMod,qCon)
                    for j in list(G_dmc_anc.nodes()):
                        if (G_dmc_anc.out_degree(j)==0 and G_dmc_anc.in_degree(j)==0):
                            G_dmc_anc.remove_node(j)
                        elif G_dmc_anc.out_degree(j)==1 and G_dmc_anc.in_degree(j)==1 and (j,j) in list(G_dmc_anc.edges):
                            G_dmc_anc.remove_node(j)
                    
                    EC_dmc_vec[str((l,m))]=normalised_ec_score(G_dmc_anc,G_anc)
                    ICS_dmc_vec[str((l,m))]=ics_score(G_dmc_anc,G_anc)
                    S3_dmc_vec[str((l,m))]=s3_score(G_dmc_anc,G_anc)
                    conservedEdges=conserved_edges(G_dmc_anc,G_anc)
                    conserved_dmc_vec[str((l,m))]=conservedEdges
                    extra_edges_dmc_vec[str((l,m))]=len(G_dmc_anc.edges)-conservedEdges
                    missed_edges_dmc_vec[str((l,m))]=len(G_anc.edges)-conservedEdges
                    network_order_dmc[str((l,m))]=len(G_dmc_anc.nodes)
                    nx.write_edgelist(G_dmc_anc,direc+"anc50_2cherry_branch"+str(branchLength)+"/dmc_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                  if myalg_align:  
                    #NF Align then apply my algorithm
                    G1_orig=copy.deepcopy(G1)
                    G2_orig=copy.deepcopy(G2)
                    G1_labelless=nx.convert_node_labels_to_integers(G1_orig)
                    G2_labelless=nx.convert_node_labels_to_integers(G2_orig)
                    P1=gene_family_partitioner(G1_labelless,orig_label=True)
                    P2=gene_family_partitioner(G2_labelless,orig_label=True)
                    print(P1,P2)
                    alignVec,mapped=NF_gene_family(G1_labelless,G2_labelless,P1,P2,32,0.8,thresh=2)
                    mapping1 = dict(alignVec)
                    mapping2=dict()
                    print(mapping1)
                    for i in list(G2_labelless.nodes):

                        if i not in mapping1.values():
                            mapping2[i]=str(i)+"_2"
                        else:
                            mapping2[i]=i
                    for i in list(G1_labelless.nodes):
                        if i not in mapping1.keys():
                            mapping1[i]=str(i)+"_1"

                    print(mapping1)
                    print(mapping2)
                    G1_mapped=nx.relabel_nodes(G1_labelless,mapping1)

                    G2_mapped=nx.relabel_nodes(G2_labelless,mapping2)

                    P1=gene_family_partitioner(G1_mapped,orig_label=True)
                    P2=gene_family_partitioner(G2_mapped,orig_label=True)
                    print(P1,P2)
                    
                    
                    t=1
                    tEC=0
                    if model=="ped_pea":
                      if algType=="gene":
                        G_anc_find1,G_anc_find2=GFAF_ped_pea(G1_mapped,G2_mapped,P1,P2,r,q,tolerance=t,toleranceEC=tEC,true_labels=True)
                    elif model=="dmc":
                      if algType=="gene":
                        G_anc_find1,G_anc_find2=GFAF_dmc(G1_mapped,G2_mapped,P1,P2,qCon,qMod,tolerance=t,toleranceEC=tEC)
                  
                    else:
                        exit
                        
                    print("ancestor found")
                    G_find_int_anc=nx.intersection(G_anc_find1,G_anc_find2)
                    print("my alg edge number",len(G_find_int_anc.edges()))
                    G_align_unint=graph_intersection_union(G_anc_find1,G_anc_find2)
                    print("my alg edge number unint",len(G_align_unint.edges()))
                    
                    
                    for j in list(G_find_int_anc.nodes()):
                        if (G_find_int_anc.out_degree(j)==0 and G_find_int_anc.in_degree(j)==0):
                            G_find_int_anc.remove_node(j)
                        elif G_find_int_anc.out_degree(j)==1 and G_find_int_anc.in_degree(j)==1 and (j,j) in list(G_find_int_anc.edges):
                            G_find_int_anc.remove_node(j)
                    mapping=dict()
                    for i in list(G_find_int_anc.nodes):
                        mapping[i]=G_anc_find1.nodes[i]['orig_label']
                    print("this is the map",mapping)
                    G_find_int_anc=nx.relabel_nodes(G_find_int_anc,mapping)

                    G_orig=copy.deepcopy(G_anc)
                    G_int_orig=copy.deepcopy(G_find_int_anc)
                    G_orig=label_conserver(G_anc)
                    G_int_orig=label_conserver(G_find_int_anc)
                    G_labelless=nx.convert_node_labels_to_integers(G_orig)
                    G_int_labelless=nx.convert_node_labels_to_integers(G_int_orig)
                    P2=gene_family_partitioner(G_labelless,orig_label=True)
                    P1=gene_family_partitioner(G_int_labelless,orig_label=True)
                    alignVec,mapped=NF_gene_family(G_int_labelless,G_labelless,P1,P2,32,0.8,thresh=2)
                    mapping1 = dict(alignVec)
                    mapping2=dict()
                    print("Alignment vector",mapping1)
                    G_int_mapped=nx.relabel_nodes(G_int_labelless,mapping1)

                    conservedEdges=conserved_edges(G_int_mapped,G_labelless)
                    conserved_align_int_vec[str((l,m))]=conservedEdges
                    extra_edges_align_int_vec[str((l,m))]=len(G_int_mapped.edges)-conservedEdges
                    missed_edges_align_int_vec[str((l,m))]=len(G_labelless.edges)-conservedEdges
                    EC_findAl_int_vec[str((l,m))]=normalised_ec_score(G_int_mapped,G_labelless)
                    ICS_findAl_int_vec[str((l,m))]=ics_score(G_int_mapped,G_labelless)
                    S3_findAl_int_vec[str((l,m))]=s3_score(G_int_mapped,G_labelless)
                    network_order_findAl_int[str((l,m))]=len(G_int_mapped.nodes)
                    

                    #conservedEdges=conserved_edges(G_find_int_anc,G_anc)
                    #conserved_align_int_vec[str((l,m))]=conservedEdges
                    #extra_edges_align_int_vec[str((l,m))]=len(G_find_int_anc.edges)-conservedEdges
                    #missed_edges_align_int_vec[str((l,m))]=len(G_anc.edges)-conservedEdges
                    #EC_findAl_int_vec[str((l,m))]=normalised_ec_score(G_find_int_anc,G_anc)
                    #ICS_findAl_int_vec[str((l,m))]=ics_score(G_find_int_anc,G_anc)
                    #S3_findAl_int_vec[str((l,m))]=s3_score(G_find_int_anc,G_anc)
                    #network_order_findAl_int[str((l,m))]=len(G_find_int_anc.nodes)

                    for j in list(G_align_unint.nodes()):
                        if (G_align_unint.out_degree(j)==0 and G_align_unint.in_degree(j)==0):
                            G_align_unint.remove_node(j)
                        elif G_align_unint.out_degree(j)==1 and G_align_unint.in_degree(j)==1 and (j,j) in list(G_align_unint.edges):
                            G_align_unint.remove_node(j)
                    
                    
                    mapping=dict()
                    for i in list(G_align_unint.nodes):
                        mapping[i]=G_anc_find1.nodes[i]['orig_label']
                    G_align_unint=nx.relabel_nodes(G_align_unint,mapping)

                    G_orig=copy.deepcopy(G_anc)
                    G_int_orig=copy.deepcopy(G_align_unint)
                    G_orig=label_conserver(G_anc)
                    G_unint_orig=label_conserver(G_align_unint)
                    G_labelless=nx.convert_node_labels_to_integers(G_orig)
                    G_unint_labelless=nx.convert_node_labels_to_integers(G_unint_orig)
                    P2=gene_family_partitioner(G_labelless,orig_label=True)
                    P1=gene_family_partitioner(G_unint_labelless,orig_label=True)
                    alignVec,mapped=NF_gene_family(G_unint_labelless,G_labelless,P1,P2,32,0.8,thresh=2)
                    mapping1 = dict(alignVec)
                    mapping2=dict()
                    print("Alignment vector",mapping1)
                    G_unint_mapped=nx.relabel_nodes(G_unint_labelless,mapping1)

                    conservedEdges=conserved_edges(G_unint_mapped,G_labelless)
                    conserved_align_unint_vec[str((l,m))]=conservedEdges
                    extra_edges_align_unint_vec[str((l,m))]=len(G_unint_mapped.edges)-conservedEdges
                    missed_edges_align_unint_vec[str((l,m))]=len(G_labelless.edges)-conservedEdges
                    EC_align_unint_vec[str((l,m))]=normalised_ec_score(G_unint_mapped,G_labelless)
                    ICS_align_unint_vec[str((l,m))]=ics_score(G_unint_mapped,G_labelless)
                    S3_align_unint_vec[str((l,m))]=s3_score(G_unint_mapped,G_labelless)
                    network_order_align_unint[str((l,m))]=len(G_unint_mapped.nodes)
                    


                    #conservedEdges=conserved_edges(G_align_unint,G_anc)
                    #conserved_align_unint_vec[str((l,m))]=conservedEdges
                    #extra_edges_align_unint_vec[str((l,m))]=len(G_align_unint.edges)-conservedEdges
                    #missed_edges_align_unint_vec[str((l,m))]=len(G_anc.edges)-conservedEdges
                    #EC_align_unint_vec[str((l,m))]=normalised_ec_score(G_align_unint,G_anc)
                    #ICS_align_unint_vec[str((l,m))]=ics_score(G_align_unint,G_anc)
                    #S3_align_unint_vec[str((l,m))]=s3_score(G_align_unint,G_anc)
                    #network_order_align_unint[str((l,m))]=len(G_align_unint.nodes)

                    for j in list(G_anc_find1.nodes()):
                        if (G_anc_find1.out_degree(j)==0 and G_anc_find1.in_degree(j)==0):
                            G_anc_find1.remove_node(j)
                        elif G_anc_find1.out_degree(j)==1 and G_anc_find1.in_degree(j)==1 and (j,j) in list(G_anc_find1.edges):
                            G_anc_find1.remove_node(j)

                    mapping=dict()
                    for i in list(G_anc_find1.nodes):
                        mapping[i]=G_anc_find1.nodes[i]['orig_label']
                    G_find1_anc=nx.relabel_nodes(G_anc_find1,mapping)
                    
                    EC_findAl1_vec[str((l,m))]=normalised_ec_score(G_find1_anc,G_anc)
                    ICS_findAl1_vec[str((l,m))]=ics_score(G_find1_anc,G_anc)
                    S3_findAl1_vec[str((l,m))]=s3_score(G_find1_anc,G_anc)
                    network_order_findAl1[str((l,m))]=len(G_find1_anc.nodes)

                    for j in list(G_anc_find2.nodes()):
                        if (G_anc_find2.out_degree(j)==0 and G_anc_find2.in_degree(j)==0):
                            G_anc_find2.remove_node(j)
                        elif G_anc_find2.out_degree(j)==1 and G_anc_find2.in_degree(j)==1 and (j,j) in list(G_anc_find2.edges):
                            G_anc_find2.remove_node(j)
                    
                    mapping=dict()
                    for i in list(G_anc_find2.nodes):
                        mapping[i]=G_anc_find2.nodes[i]['orig_label']
                    G_find2_anc=nx.relabel_nodes(G_anc_find2,mapping)

                    EC_findAl2_vec[str((l,m))]=normalised_ec_score(G_find2_anc,G_anc)
                    ICS_findAl2_vec[str((l,m))]=ics_score(G_find2_anc,G_anc)
                    S3_findAl2_vec[str((l,m))]=s3_score(G_find2_anc,G_anc)
                    network_order_findAl2[str((l,m))]=len(G_find2_anc.nodes)

                    nx.write_edgelist(G_find_int_anc,direc+"anc50_2cherry_branch"+str(branchLength)+"/myalg_align_int_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                    nx.write_edgelist(G_align_unint,direc+"anc50_2cherry_branch"+str(branchLength)+"/myalg_align_unint_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                    nx.write_edgelist(G_find1_anc,direc+"anc50_2cherry_branch"+str(branchLength)+"/myalgG"+str(l)+"_align_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                    nx.write_edgelist(G_find2_anc,direc+"anc50_2cherry_branch"+str(branchLength)+"/myalgG"+str(m)+"_align_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                  if noalg_align:  
                    #Align then intersect
                    G1_orig=copy.deepcopy(G1)
                    G2_orig=copy.deepcopy(G2)
                    G1_labelless=nx.convert_node_labels_to_integers(G1_orig)
                    G2_labelless=nx.convert_node_labels_to_integers(G2_orig)
                    P1=gene_family_partitioner(G1_labelless,orig_label=True)
                    P2=gene_family_partitioner(G2_labelless,orig_label=True)
                    print(P1,P2)
                    alignVec,mapped=NF_gene_family(G1_labelless,G2_labelless,P1,P2,32,0.8,thresh=2)
                    mapping1 = dict(alignVec)
                    mapping2=dict()
                    print(mapping1)
                    for i in list(G2_labelless.nodes):

                        if i not in mapping1.values():
                            mapping2[i]=str(i)+"_2"
                        else:
                            mapping2[i]=i
                    for i in list(G1_labelless.nodes):
                        if i not in mapping1.keys():
                            mapping1[i]=str(i)+"_1"

                    print("NF ANC ALIGNMENT MAP G1",mapping1)
                    print("NF ANC ALIGNMENT MAP G2",mapping2)
                    G1_mapped=nx.relabel_nodes(G1_labelless,mapping1)

                    G2_mapped=nx.relabel_nodes(G2_labelless,mapping2)

                    P1=gene_family_partitioner(G1_mapped,orig_label=True)
                    P2=gene_family_partitioner(G2_mapped,orig_label=True)
                    print("NF ANC P1 and P2",P1,P2)
                    G_intersect=nx.intersection(G1_mapped,G2_mapped)

                    for j in list(G_intersect.nodes()):
                        if (G_intersect.out_degree(j)==0 and G_intersect.in_degree(j)==0):
                            G_intersect.remove_node(j)
                        elif G_intersect.out_degree(j)==1 and G_intersect.in_degree(j)==1 and (j,j) in list(G_intersect.edges):
                            G_intersect.remove_node(j)

                    mapping=dict()
                    for i in list(G_intersect.nodes):
                        mapping[i]=G1_mapped.nodes[i]['orig_label']
                    G_nf_anc=nx.relabel_nodes(G_intersect,mapping)
                    print("NF ANC BACK TO ANC",mapping)

                    
                    G_orig=copy.deepcopy(G_anc)
                    G_nf_orig=copy.deepcopy(G_nf_anc)
                    G_orig=label_conserver(G_anc)
                    G_nf_orig=label_conserver(G_nf_anc)
                    G_labelless=nx.convert_node_labels_to_integers(G_orig)
                    G_nf_labelless=nx.convert_node_labels_to_integers(G_nf_orig)
                    P2=gene_family_partitioner(G_labelless,orig_label=True)
                    P1=gene_family_partitioner(G_nf_labelless,orig_label=True)
                    alignVec,mapped=NF_gene_family(G_nf_labelless,G_labelless,P1,P2,32,0.8,thresh=2)
                    mapping1 = dict(alignVec)
                    mapping2=dict()
                    print("Alignment vector",mapping1)
                    G_nf_mapped=nx.relabel_nodes(G_nf_labelless,mapping1)

                    conservedEdges=conserved_edges(G_nf_mapped,G_labelless)
                    conserved_nf_vec[str((l,m))]=conservedEdges
                    extra_edges_nf_vec[str((l,m))]=len(G_nf_mapped.edges)-conservedEdges
                    missed_edges_nf_vec[str((l,m))]=len(G_labelless.edges)-conservedEdges
                    EC_nf_vec[str((l,m))]=normalised_ec_score(G_nf_mapped,G_labelless)
                    ICS_nf_vec[str((l,m))]=ics_score(G_nf_mapped,G_labelless)
                    S3_nf_vec[str((l,m))]=s3_score(G_nf_mapped,G_labelless)
                    network_order_nf[str((l,m))]=len(G_nf_mapped.nodes)
                    


                    #conservedEdges=conserved_edges(G_nf_anc,G_anc)
                    #conserved_nf_vec[str((l,m))]=conservedEdges
                    #extra_edges_nf_vec[str((l,m))]=len(G_nf_anc.edges)-conservedEdges
                    #missed_edges_nf_vec[str((l,m))]=len(G_anc.edges)-conservedEdges
                    #EC_nf_vec[str((l,m))]=normalised_ec_score(G_nf_anc,G_anc)
                    #ICS_nf_vec[str((l,m))]=ics_score(G_nf_anc,G_anc)
                    #S3_nf_vec[str((l,m))]=s3_score(G_nf_anc,G_anc)
                    #network_order_nf[str((l,m))]=len(G_nf_anc.nodes)
                    nx.write_edgelist(G_nf_anc,direc+"anc50_2cherry_branch"+str(branchLength)+"/nf_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                    
                    #Align then unint
                    G1_orig=copy.deepcopy(G1)
                    G2_orig=copy.deepcopy(G2)
                    G1_labelless=nx.convert_node_labels_to_integers(G1_orig)
                    G2_labelless=nx.convert_node_labels_to_integers(G2_orig)
                    P1=gene_family_partitioner(G1_labelless,orig_label=True)
                    P2=gene_family_partitioner(G2_labelless,orig_label=True)
                    print(P1,P2)
                    alignVec,mapped=NF_gene_family(G1_labelless,G2_labelless,P1,P2,32,0.8,thresh=2)
                    mapping1 = dict(alignVec)
                    mapping2=dict()
                    print(mapping1)
                    for i in list(G2_labelless.nodes):

                        if i not in mapping1.values():
                            mapping2[i]=str(i)+"_2"
                        else:
                            mapping2[i]=i
                    for i in list(G1_labelless.nodes):
                        if i not in mapping1.keys():
                            mapping1[i]=str(i)+"_1"

                    print("NF ANC ALIGNMENT MAP G1",mapping1)
                    print("NF ANC ALIGNMENT MAP G2",mapping2)
                    G1_mapped=nx.relabel_nodes(G1_labelless,mapping1)

                    G2_mapped=nx.relabel_nodes(G2_labelless,mapping2)

                    P1=gene_family_partitioner(G1_mapped,orig_label=True)
                    P2=gene_family_partitioner(G2_mapped,orig_label=True)
                    print("NF ANC P1 and P2",P1,P2)
                    G_unint=graph_intersection_union(G1_mapped,G2_mapped)

                    for j in list(G_unint.nodes()):
                        if (G_unint.out_degree(j)==0 and G_unint.in_degree(j)==0):
                            G_unint.remove_node(j)
                        elif G_unint.out_degree(j)==1 and G_unint.in_degree(j)==1 and (j,j) in list(G_unint.edges):
                            G_unint.remove_node(j)

                    mapping=dict()
                    for i in list(G_unint.nodes):
                        mapping[i]=G1_mapped.nodes[i]['orig_label']
                    G_nf_unint_anc=nx.relabel_nodes(G_unint,mapping)

                    
                    G_orig=copy.deepcopy(G_anc)
                    G_nf_orig=copy.deepcopy(G_nf_unint_anc)
                    G_orig=label_conserver(G_anc)
                    G_nf_unint_orig=label_conserver(G_nf_anc)
                    G_labelless=nx.convert_node_labels_to_integers(G_orig)
                    G_nf_unint_labelless=nx.convert_node_labels_to_integers(G_nf_unint_orig)
                    P2=gene_family_partitioner(G_labelless,orig_label=True)
                    P1=gene_family_partitioner(G_nf_unint_labelless,orig_label=True)
                    alignVec,mapped=NF_gene_family(G_nf_unint_labelless,G_labelless,P1,P2,32,0.8,thresh=2)
                    mapping1 = dict(alignVec)
                    mapping2=dict()
                    print("Alignment vector",mapping1)
                    G_nf_unint_mapped=nx.relabel_nodes(G_nf_unint_labelless,mapping1)

                    conservedEdges=conserved_edges(G_nf_unint_mapped,G_labelless)
                    conserved_nf_unint_vec[str((l,m))]=conservedEdges
                    extra_edges_nf_unint_vec[str((l,m))]=len(G_nf_unint_mapped.edges)-conservedEdges
                    missed_edges_nf_unint_vec[str((l,m))]=len(G_labelless.edges)-conservedEdges
                    EC_nf_unint_vec[str((l,m))]=normalised_ec_score(G_nf_unint_mapped,G_labelless)
                    ICS_nf_unint_vec[str((l,m))]=ics_score(G_nf_unint_mapped,G_labelless)
                    S3_nf_unint_vec[str((l,m))]=s3_score(G_nf_mapped,G_labelless)
                    network_order_nf_unint[str((l,m))]=len(G_nf_mapped.nodes)
                    #print("NF ANC BACK TO ANC",mapping)
                    #conservedEdges=conserved_edges(G_nf_unint_anc,G_anc)
                    #conserved_nf_unint_vec[str((l,m))]=conservedEdges
                    #extra_edges_nf_unint_vec[str((l,m))]=len(G_nf_unint_anc.edges)-conservedEdges
                    #missed_edges_nf_unint_vec[str((l,m))]=len(G_anc.edges)-conservedEdges
                    #EC_nf_unint_vec[str((l,m))]=normalised_ec_score(G_nf_unint_anc,G_anc)
                    #ICS_nf_unint_vec[str((l,m))]=ics_score(G_nf_unint_anc,G_anc)
                    #S3_nf_unint_vec[str((l,m))]=s3_score(G_nf_unint_anc,G_anc)
                    #network_order_nf_unint[str((l,m))]=len(G_nf_unint_anc.nodes)
                    nx.write_edgelist(G_nf_unint_anc,direc+"anc50_2cherry_branch"+str(branchLength)+"/nf_unint_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                  if noalg:  
                    #True Label Intersection
                    G1_orig=copy.deepcopy(G1)
                    G2_orig=copy.deepcopy(G2)
                    #G1_orig=nx.relabel_nodes(G1_orig,mapper1)
                    #G2_orig=nx.relabel_nodes(G2_orig,mapper2)
                    
                    G_intersect=nx.intersection(G1_orig,G2_orig)
                    print("intersect edge number",len(G_intersect.edges()))

                    for j in list(G_intersect.nodes()):
                        if (G_intersect.out_degree(j)==0 and G_intersect.in_degree(j)==0):
                            G_intersect.remove_node(j)
                        elif G_intersect.out_degree(j)==1 and G_intersect.in_degree(j)==1 and (j,j) in list(G_intersect.edges):
                            G_intersect.remove_node(j)
                    mapping=dict()
                    for i in list(G_intersect.nodes):
                        mapping[i]=G1_orig.nodes[i]['orig_label']
                    G_intersect=nx.relabel_nodes(G_intersect,mapping)

                    G_orig_anc=G_intersect
                    
                    EC_orig_vec[str((l,m))]=normalised_ec_score(G_orig_anc,G_anc)
                    ICS_orig_vec[str((l,m))]=ics_score(G_orig_anc,G_anc)
                    S3_orig_vec[str((l,m))]=s3_score(G_orig_anc,G_anc)
                    conservedEdges=conserved_edges(G_orig_anc,G_anc)
                    conserved_orig_vec[str((l,m))]=conservedEdges
                    extra_edges_orig_vec[str((l,m))]=len(G_orig_anc.edges)-conservedEdges
                    missed_edges_orig_vec[str((l,m))]=len(G_anc.edges)-conservedEdges
                    network_order_orig[str((l,m))]=len(G_orig_anc.nodes)

                    G_unint_anc=graph_intersection_union(G1_orig,G2_orig)
                    mapping=dict()
                    for i in list(G_unint_anc.nodes):
                        mapping[i]=G1_orig.nodes[i]['orig_label']
                    G_unint_anc=nx.relabel_nodes(G_unint_anc,mapping)
                    EC_unint_vec[str((l,m))]=normalised_ec_score(G_unint_anc,G_anc)
                    ICS_unint_vec[str((l,m))]=ics_score(G_unint_anc,G_anc)
                    S3_unint_vec[str((l,m))]=s3_score(G_unint_anc,G_anc)
                    conservedEdges=conserved_edges(G_unint_anc,G_anc)
                    conserved_unint_vec[str((l,m))]=conservedEdges
                    extra_edges_unint_vec[str((l,m))]=len(G_unint_anc.edges)-conservedEdges
                    missed_edges_unint_vec[str((l,m))]=len(G_anc.edges)-conservedEdges
                    network_order_unint[str((l,m))]=len(G_unint_anc.nodes)
                    
                    nx.write_edgelist(G_orig_anc,direc+"anc50_2cherry_branch"+str(branchLength)+"/intersect_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                    nx.write_edgelist(G_unint_anc,direc+"anc50_2cherry_branch"+str(branchLength)+"/unint_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                    print("g_anc true edge number",len(G_anc.edges())) 
                    
  pairs_vec=[]

  EC_nf=[]
  EC_nf_unint=[]
  EC_dmc=[]
  EC_orig=[]
  EC_unint=[]
  EC_find_int=[]
  EC_findAl_int=[]
  EC_true_unint=[]
  EC_align_unint=[]

  S3_nf=[]
  S3_nf_unint=[]
  S3_dmc=[]
  S3_orig=[]
  S3_unint=[]
  S3_find_int=[]
  S3_findAl_int=[]
  S3_true_unint=[]
  S3_align_unint=[]

  net_order_nf=[]
  net_order_nf_unint=[]
  net_order_dmc=[]
  net_order_orig=[]
  net_order_find1=[]
  net_order_find2=[]
  net_order_find_int=[]
  net_order_findAl_int=[]
  net_order_true_unint=[]
  net_order_align_unint=[]
  net_order_unint=[]

  conserved_nf=[]
  conserved_nf_unint=[]
  conserved_dmc=[]
  conserved_orig=[]
  conserved_unint=[]
  conserved_true_unint=[]
  conserved_find_int=[]

  extra_edges_nf=[]
  extra_edges_nf_unint=[]
  extra_edges_dmc=[]
  extra_edges_orig=[]
  extra_edges_unint=[]
  extra_edges_true_unint=[]
  extra_edges_find_int=[]

  missed_edges_nf=[]
  missed_edges_nf_unint=[]
  missed_edges_dmc=[]
  missed_edges_orig=[]
  missed_edges_unint=[]
  missed_edges_true_unint=[]
  missed_edges_find_int=[]

  conserved_align_int=[]
  conserved_align_unint=[]
  extra_edges_align_int=[]
  extra_edges_align_unint=[]
  missed_edges_align_int=[]
  missed_edges_align_unint=[]

  font = {'family' : 'normal',
          'weight' : 'normal',

          'size'   : 18}


  plt.rc('font', **font)
  for l in leafGraphs:
      for m in leafGraphs:
          if m>l:
              pairs_vec.append(str((l,m)))
              if myalg:
                EC_find_int.append(EC_find_int_vec[str((l,m))])
                EC_true_unint.append(EC_true_unint_vec[str((l,m))])

                S3_true_unint.append(S3_true_unint_vec[str((l,m))])
                S3_find_int.append(S3_find_int_vec[str((l,m))])

                conserved_true_unint.append(conserved_true_unint_vec[str((l,m))])
                conserved_find_int.append(conserved_find_int_vec[str((l,m))])

                
                extra_edges_true_unint.append(extra_edges_true_unint_vec[str((l,m))])
                extra_edges_find_int.append(extra_edges_find_int_vec[str((l,m))])

                missed_edges_true_unint.append(missed_edges_true_unint_vec[str((l,m))])
                missed_edges_find_int.append(missed_edges_find_int_vec[str((l,m))])

                net_order_find1.append(network_order_find1[str((l,m))])
                net_order_find2.append(network_order_find2[str((l,m))])
                net_order_find_int.append(network_order_find_int[str((l,m))])
                net_order_true_unint.append(network_order_true_unint[str((l,m))])
              if myalg_align:
                EC_align_unint.append(EC_align_unint_vec[str((l,m))])
                EC_findAl_int.append(EC_findAl_int_vec[str((l,m))])

                S3_align_unint.append(S3_align_unint_vec[str((l,m))])
                S3_findAl_int.append(S3_findAl_int_vec[str((l,m))])

                conserved_align_int.append(conserved_align_int_vec[str((l,m))])
                conserved_align_unint.append(conserved_align_unint_vec[str((l,m))])

                extra_edges_align_int.append(extra_edges_align_int_vec[str((l,m))])
                extra_edges_align_unint.append(extra_edges_align_unint_vec[str((l,m))])

                missed_edges_align_int.append(missed_edges_align_int_vec[str((l,m))])
                missed_edges_align_unint.append(missed_edges_align_unint_vec[str((l,m))])

                net_order_findAl_int.append(network_order_findAl_int[str((l,m))])
                net_order_align_unint.append(network_order_align_unint[str((l,m))])
              if noalg:
                EC_orig.append(EC_orig_vec[str((l,m))])
                EC_unint.append(EC_unint_vec[str((l,m))])

                S3_orig.append(S3_orig_vec[str((l,m))])
                S3_unint.append(S3_unint_vec[str((l,m))])

                conserved_orig.append(conserved_orig_vec[str((l,m))])
                conserved_unint.append(conserved_unint_vec[str((l,m))])

                extra_edges_orig.append(extra_edges_orig_vec[str((l,m))])
                extra_edges_unint.append(extra_edges_unint_vec[str((l,m))])

                missed_edges_orig.append(missed_edges_orig_vec[str((l,m))])
                missed_edges_unint.append(missed_edges_unint_vec[str((l,m))])

                
                net_order_orig.append(network_order_orig[str((l,m))])
              if noalg_align:
                EC_nf.append(EC_nf_vec[str((l,m))])
                EC_nf_unint.append(EC_nf_unint_vec[str((l,m))])

                S3_nf.append(S3_nf_vec[str((l,m))])
                S3_nf_unint.append(S3_nf_unint_vec[str((l,m))])

                net_order_nf.append(network_order_nf[str((l,m))])
                net_order_nf_unint.append(network_order_nf_unint[str((l,m))])

                missed_edges_nf.append(missed_edges_nf_vec[str((l,m))])
                missed_edges_nf_unint.append(missed_edges_nf_unint_vec[str((l,m))])

                extra_edges_nf.append(extra_edges_nf_vec[str((l,m))])
                extra_edges_nf_unint.append(extra_edges_nf_unint_vec[str((l,m))])

                conserved_nf.append(conserved_nf_vec[str((l,m))])
                conserved_nf_unint.append(conserved_nf_unint_vec[str((l,m))])

                
                net_order_unint.append(network_order_unint[str((l,m))])
              if zhu_nakleh:
                EC_dmc.append(EC_dmc_vec[str((l,m))])
                S3_dmc.append(S3_dmc_vec[str((l,m))])
                net_order_dmc.append(network_order_dmc[str((l,m))])
                conserved_dmc.append(conserved_dmc_vec[str((l,m))])
                extra_edges_dmc.append(extra_edges_dmc_vec[str((l,m))])
                missed_edges_dmc.append(missed_edges_dmc_vec[str((l,m))])
              
  print(pairs_vec)
  plt.figure(figsize =(12, 8))
  barWidth = 0.08
  fig = plt.subplots(figsize =(12, 8))
  # Set position of bar on X axis
  br1 = np.arange(len(EC_dmc))
  br2 = [x + barWidth for x in br1]
  br3 = [x + barWidth for x in br2]
  br4 = [x + barWidth for x in br3]
  br5 = [x + barWidth for x in br4]
  br6 = [x + barWidth for x in br5]
  br7 = [x + barWidth for x in br6]
  br8 = [x + barWidth for x in br7]
  br9 = [x + barWidth for x in br8]


  # Make the plot

  plt.bar(br1, EC_dmc, color ='g', width = barWidth,label ="Zhu-Nakleh Reconstruction")
  plt.bar(br2, EC_orig, color ='b', width = barWidth,label ="True Label Int")
  plt.bar(br3, EC_unint, color ='c', width = barWidth,label ="True Label Unint")
  plt.bar(br4, EC_find_int, color ='y', width = barWidth,label ="My Algorithm (True Label Int)")
  plt.bar(br5, EC_true_unint, color ='m', width = barWidth,label ="My Algorithm (True Label Unint)")
  plt.bar(br6, EC_nf, color ='r', width = barWidth,label ="Align then Int")
  plt.bar(br7, EC_nf_unint, color ='darkorchid', width = barWidth,label ="Align then Unint")
  plt.bar(br8, EC_findAl_int, color ='k', width = barWidth,label ="My Algorithm (Align Int)")
  plt.bar(br9, EC_align_unint, color ='bisque', width = barWidth,label ="My Algorithm (Align Unint)")


  # Adding Xticks
  plt.xlabel('Leaf Network Pair')
  plt.ylabel('EC Score of Predicted and Actual Ancestor')
  plt.xticks([r + barWidth+0.16 for r in range(len(EC_dmc))],
          pairs_vec)

  # Put a legend to the right of the current axis
  plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

  plt.savefig(direc+"anc50_2cherry_branch"+str(branchLength)+"/EC_scores_branchlength"+str(branchLength)+".png",bbox_inches="tight")

  plt.figure(figsize =(12, 8))
  barWidth = 0.08
  fig = plt.subplots(figsize =(12, 8))
  # Set position of bar on X axis
  br1 = np.arange(len(S3_dmc))
  br2 = [x + barWidth for x in br1]
  br3 = [x + barWidth for x in br2]
  br4 = [x + barWidth for x in br3]
  br5 = [x + barWidth for x in br4]
  br6 = [x + barWidth for x in br5]
  br7 = [x + barWidth for x in br6]
  br8 = [x + barWidth for x in br7]
  br9 = [x + barWidth for x in br8]


  # Make the plot

  plt.bar(br1, S3_dmc, color ='g', width = barWidth,label ="Zhu-Nakleh Reconstruction")
  plt.bar(br2, S3_orig, color ='b', width = barWidth,label ="True Label Int")
  plt.bar(br3, S3_unint, color ='c', width = barWidth,label ="True Label Unint")
  plt.bar(br4, S3_find_int, color ='y', width = barWidth,label ="My Algorithm (True Label Int)")
  plt.bar(br5, S3_true_unint, color ='m', width = barWidth,label ="My Algorithm (True Label Unint)")
  plt.bar(br6, S3_nf, color ='r', width = barWidth,label ="Align then Int")
  plt.bar(br7, S3_nf_unint, color ='darkorchid', width = barWidth,label ="Align then Unint")
  plt.bar(br8, S3_findAl_int, color ='k', width = barWidth,label ="My Algorithm (Align Int)")
  plt.bar(br9, S3_align_unint, color ='bisque', width = barWidth,label ="My Algorithm (Align Unint)")


  # Adding Xticks
  plt.xlabel('Leaf Network Pair')
  plt.ylabel('S3 Score of Predicted and Actual Ancestor')
  plt.xticks([r + barWidth+0.16 for r in range(len(S3_dmc))],
          pairs_vec)

  plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
  plt.savefig(direc+"anc50_2cherry_branch"+str(branchLength)+"/S3_scores_branchlength"+str(branchLength)+".png",bbox_inches='tight')

  plt.figure(figsize =(12, 8))
  barWidth = 0.08
  fig = plt.subplots(figsize =(12, 8))
  # Set position of bar on X axis
  br1 = np.arange(len(net_order_dmc))
  br2 = [x + barWidth for x in br1]
  br3 = [x + barWidth for x in br2]
  br4 = [x + barWidth for x in br3]
  br5 = [x + barWidth for x in br4]
  br6 = [x + barWidth for x in br5]
  br7 = [x + barWidth for x in br6]
  br8 = [x + barWidth for x in br7]
  br9 = [x + barWidth for x in br8]
  br10=[x + barWidth for x in br9]
  br11=[x + barWidth for x in br10]


  # Make the plot

  plt.bar(br1, net_order_dmc, color ='g', width = barWidth,label ="Zhu-Nakleh Reconstruction")
  plt.bar(br2, net_order_orig, color ='b', width = barWidth,label ="True Label Int")
  plt.bar(br3, net_order_unint, color ='c', width = barWidth,label ="True Label Unint")
  plt.bar(br4, net_order_find_int, color ='y', width = barWidth,label ="My Algorithm (True Label Int)")
  plt.bar(br5, net_order_true_unint, color ='m', width = barWidth,label ="My Algorithm (True Label Unint)")
  plt.bar(br6, net_order_find1, color ='grey', width = barWidth,label ="My Algorithm G1")
  plt.bar(br7, net_order_find2, color ='gold', width = barWidth,label ="My Algorithm G2")
  plt.bar(br8, net_order_nf, color ='r', width = barWidth,label ="Align then Int")
  plt.bar(br9, net_order_nf_unint, color ='darkorchid', width = barWidth,label ="Align then Unint")
  plt.bar(br10, net_order_findAl_int, color ='k', width = barWidth,label ="My Algorithm (Align Int)")
  plt.bar(br11, net_order_align_unint, color ='bisque', width = barWidth,label ="My Algorithm (Align Unint)")

  # Adding Xticks
  plt.xlabel('Leaf Network Pair')
  plt.ylabel('Network Order of Predicted and Actual Ancestor')
  plt.xticks([r + barWidth+0.16 for r in range(len(net_order_dmc))],
          pairs_vec)

  plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
  plt.savefig(direc+"anc50_2cherry_branch"+str(branchLength)+"/net_order_scores_branchlength"+str(branchLength)+".png",bbox_inches='tight')

  plt.figure(figsize =(12, 8))
  barWidth = 0.08
  fig = plt.subplots(figsize =(12, 8))
  # Set position of bar on X axis
  br1 = np.arange(len(conserved_dmc))
  br2 = [x + barWidth for x in br1]
  br3 = [x + barWidth for x in br2]
  br4 = [x + barWidth for x in br3]
  br5 = [x + barWidth for x in br4]
  br6 = [x + barWidth for x in br5]
  br7 = [x + barWidth for x in br6]
  br8 = [x + barWidth for x in br7]
  br9 = [x + barWidth for x in br8]


  # Make the plot

  plt.bar(br1, conserved_dmc, color ='g', width = barWidth,label ="Zhu-Nakleh Reconstruction")
  plt.bar(br2, conserved_orig, color ='b', width = barWidth,label ="True Label Int")
  plt.bar(br3, conserved_unint, color ='c', width = barWidth,label ="True Label Unint")
  plt.bar(br4, conserved_find_int, color ='y', width = barWidth,label ="My Algorithm (True Label Int)")
  plt.bar(br5, conserved_true_unint, color ='m', width = barWidth,label ="My Algorithm (True Label Unint)")
  plt.bar(br6, conserved_nf, color ='r', width = barWidth,label ="Align then Int")
  plt.bar(br7, conserved_nf_unint, color ='darkorchid', width = barWidth,label ="Align then Unint")
  plt.bar(br8, conserved_align_int, color ='k', width = barWidth,label ="My Algorithm (Align Int)")
  plt.bar(br9, conserved_align_unint, color ='bisque', width = barWidth,label ="My Algorithm (Align Unint)")


  # Adding Xticks
  plt.xlabel('Leaf Network Pair')
  plt.ylabel('No. of Correct Edges in Predicted Ancestor')
  plt.xticks([r + barWidth+0.16 for r in range(len(conserved_dmc))],
          pairs_vec)

  # Put a legend to the right of the current axis
  plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

  plt.savefig(direc+"anc50_2cherry_branch"+str(branchLength)+"/conserved_edges_branchlength"+str(branchLength)+".png",bbox_inches="tight")


  plt.figure(figsize =(12, 8))

  barWidth = 0.08
  fig = plt.subplots(figsize =(12, 8))
  # Set position of bar on X axis
  br1 = np.arange(len(conserved_dmc))
  br2 = [x + barWidth for x in br1]
  br3 = [x + barWidth for x in br2]
  br4 = [x + barWidth for x in br3]
  br5 = [x + barWidth for x in br4]
  br6 = [x + barWidth for x in br5]
  br7 = [x + barWidth for x in br6]
  br8 = [x + barWidth for x in br7]
  br9 = [x + barWidth for x in br8]


  # Make the plot

  plt.bar(br1, extra_edges_dmc, color ='g', width = barWidth,label ="Zhu-Nakleh Reconstruction")
  plt.bar(br2, extra_edges_orig, color ='b', width = barWidth,label ="True Label Int")
  plt.bar(br3, extra_edges_unint, color ='c', width = barWidth,label ="True Label Unint")
  plt.bar(br4, extra_edges_find_int, color ='y', width = barWidth,label ="My Algorithm (True Label Int)")
  plt.bar(br5, extra_edges_true_unint, color ='m', width = barWidth,label ="My Algorithm (True Label Unint)")
  plt.bar(br6, extra_edges_nf, color ='r', width = barWidth,label ="Align then Int")
  plt.bar(br7, extra_edges_nf_unint, color ='darkorchid', width = barWidth,label ="Align then Unint")
  plt.bar(br8, extra_edges_align_int, color ='k', width = barWidth,label ="My Algorithm (Align Int)")
  plt.bar(br9, extra_edges_align_unint, color ='bisque', width = barWidth,label ="My Algorithm (Align Unint)")


  # Adding Xticks
  plt.xlabel('Leaf Network Pair')
  plt.ylabel('No. of Extra Edges in Predicted Ancestor')
  plt.xticks([r + barWidth+0.16 for r in range(len(conserved_dmc))],
          pairs_vec)

  # Put a legend to the right of the current axis
  plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

  plt.savefig(direc+"anc50_2cherry_branch"+str(branchLength)+"/extra_edges_branchlength"+str(branchLength)+".png",bbox_inches="tight")


  plt.figure(figsize =(12, 8))

  barWidth = 0.08
  fig = plt.subplots(figsize =(12, 8))
  # Set position of bar on X axis
  br1 = np.arange(len(conserved_dmc))
  br2 = [x + barWidth for x in br1]
  br3 = [x + barWidth for x in br2]
  br4 = [x + barWidth for x in br3]
  br5 = [x + barWidth for x in br4]
  br6 = [x + barWidth for x in br5]
  br7 = [x + barWidth for x in br6]
  br8 = [x + barWidth for x in br7]
  br9 = [x + barWidth for x in br8]


  # Make the plot

  plt.bar(br1, missed_edges_dmc, color ='g', width = barWidth,label ="Zhu-Nakleh Reconstruction")
  plt.bar(br2, missed_edges_orig, color ='b', width = barWidth,label ="True Label Int")
  plt.bar(br3, missed_edges_unint, color ='c', width = barWidth,label ="True Label Unint")
  plt.bar(br4, missed_edges_find_int, color ='y', width = barWidth,label ="My Algorithm (True Label Int)")
  plt.bar(br5, missed_edges_true_unint, color ='m', width = barWidth,label ="My Algorithm (True Label Unint)")
  plt.bar(br6, missed_edges_nf, color ='r', width = barWidth,label ="Align then Int")
  plt.bar(br7, missed_edges_nf, color ='darkorchid', width = barWidth,label ="Align then Unint")
  plt.bar(br8, missed_edges_align_int, color ='k', width = barWidth,label ="My Algorithm (Align Int)")
  plt.bar(br9, missed_edges_align_unint, color ='bisque', width = barWidth,label ="My Algorithm (Align Unint)")


  # Adding Xticks
  plt.xlabel('Leaf Network Pair')
  plt.ylabel('No. Edges in True Ancestor Missed By Predicted Ancestor')
  plt.xticks([r + barWidth+0.16 for r in range(len(conserved_dmc))],
          pairs_vec)

  # Put a legend to the right of the current axis
  plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

  plt.savefig(direc+"anc50_2cherry_branch"+str(branchLength)+"/missed_edges_branchlength"+str(branchLength)+".png",bbox_inches="tight")
  write_list_to_file([algType],"test_datasets_ancestral/regulatory_test_ped_pea/anc50_2cherry_branch"+str(branchLength)+"/recent_run")

def to_divide_by_n_or_to_not_divide_by_n(direc="test_datasets_ancestral/varying_r_ancestral/",r=2,q=0.4):
  
  
  G1= nx.read_edgelist(direc+str(r)+"_1"+".txt",create_using=nx.DiGraph)
  G2= nx.read_edgelist(direc+str(r)+"_2"+".txt",create_using=nx.DiGraph)
  G_anc=nx.read_edgelist(direc+str(r)+"_anc"+".txt",create_using=nx.DiGraph)

  EC_find1_vec=dict()
  ICS_find1_vec=dict()
  S3_find1_vec=dict()
  network_order_find1=dict()
  EC_find2_vec=dict()
  ICS_find2_vec=dict()
  S3_find2_vec=dict()
  network_order_find2=dict()
  EC_find_int_vec=dict()
  ICS_find_int_vec=dict()
  S3_find_int_vec=dict()
  network_order_find_int=dict()

  EC_true_unint_vec=dict()
  ICS_true_unint_vec=dict()
  S3_true_unint_vec=dict()
  network_order_true_unint=dict()



  conserved_true_unint_vec=dict()
  conserved_find_int_vec=dict()


  extra_edges_true_unint_vec=dict()
  extra_edges_find_int_vec=dict()


  missed_edges_true_unint_vec=dict()
  missed_edges_find_int_vec=dict()
  for l in [G1]:
      for m in [G2]:
          if True:

              G_anc=label_conserver(G_anc)
              print("g_anc true edge number",len(G_anc.edges()))  
              G1=label_conserver(G1)
              G2=label_conserver(G2)
              
              for k in range(1):
                  if True:

                    G1_orig=copy.deepcopy(G1)
                    G2_orig=copy.deepcopy(G2)

                    t=1
                    tEC=0
               
                    G_anc_find1,G_anc_find2=GFAF_ped_pea(G1_orig,G2_orig,r,q,tolerance=t,toleranceEC=tEC,true_labels=True,divide_by_n=False)
                   
                    print("ancestor found")
                    G_find_int_anc=nx.intersection(G_anc_find1,G_anc_find2)
                    print("my alg edge number int",len(G_find_int_anc.edges()))
                    G_true_unint=graph_intersection_union(G_anc_find1,G_anc_find2)
                    print("my alg edge number unint",len(G_true_unint.edges()))
                    mapping=dict()
                    for i in list(G_find_int_anc.nodes):
                        mapping[i]=G_anc_find1.nodes[i]['orig_label']

                    G_true_unint=nx.relabel_nodes(G_true_unint,mapping)    
                    EC_true_unint_vec[0]=normalised_ec_score(G_true_unint,G_anc)
                    ICS_true_unint_vec[0]=ics_score(G_true_unint,G_anc)
                    S3_true_unint_vec[0]=s3_score(G_true_unint,G_anc)
                    conservedEdges=conserved_edges(G_true_unint,G_anc)
                    conserved_true_unint_vec[0]=conservedEdges
                    extra_edges_true_unint_vec[0]=len(G_true_unint.edges)-conservedEdges
                    missed_edges_true_unint_vec[0]=len(G_anc.edges)-conservedEdges
                    network_order_true_unint[0]=len(G_true_unint.nodes)
                    for j in list(G_find_int_anc.nodes()):
                        if (G_find_int_anc.out_degree(j)==0 and G_find_int_anc.in_degree(j)==0):
                            G_find_int_anc.remove_node(j)
                        elif G_find_int_anc.out_degree(j)==1 and G_find_int_anc.in_degree(j)==1 and (j,j) in list(G_find_int_anc.edges):
                            G_find_int_anc.remove_node(j)
                    
                    
                    
                    G_find_int_anc=nx.relabel_nodes(G_find_int_anc,mapping)

                    EC_find_int_vec[0]=normalised_ec_score(G_find_int_anc,G_anc)
                    ICS_find_int_vec[0]=ics_score(G_find_int_anc,G_anc)
                    S3_find_int_vec[0]=s3_score(G_find_int_anc,G_anc)
                    conservedEdges=conserved_edges(G_find_int_anc,G_anc)
                    conserved_find_int_vec[0]=conservedEdges
                    extra_edges_find_int_vec[0]=len(G_find_int_anc.edges)-conservedEdges
                    missed_edges_find_int_vec[0]=len(G_anc.edges)-conservedEdges
                    network_order_find_int[0]=len(G_find_int_anc.nodes)

                    for j in list(G_anc_find1.nodes()):
                        if (G_anc_find1.out_degree(j)==0 and G_anc_find1.in_degree(j)==0):
                            G_anc_find1.remove_node(j)
                        elif G_anc_find1.out_degree(j)==1 and G_anc_find1.in_degree(j)==1 and (j,j) in list(G_anc_find1.edges):
                            G_anc_find1.remove_node(j)

                    mapping=dict()
                    for i in list(G_anc_find1.nodes):
                        mapping[i]=G_anc_find1.nodes[i]['orig_label']
                    G_find1_anc=nx.relabel_nodes(G_anc_find1,mapping)
                    
                    EC_find1_vec[0]=normalised_ec_score(G_find1_anc,G_anc)
                    ICS_find1_vec[0]=ics_score(G_find1_anc,G_anc)
                    S3_find1_vec[0]=s3_score(G_find1_anc,G_anc)
                    network_order_find1[0]=len(G_find1_anc.nodes)
                    
                    for j in list(G_anc_find2.nodes()):
                        if (G_anc_find2.out_degree(j)==0 and G_anc_find2.in_degree(j)==0):
                            G_anc_find2.remove_node(j)
                        elif G_anc_find2.out_degree(j)==1 and G_anc_find2.in_degree(j)==1 and (j,j) in list(G_anc_find2.edges):
                            G_anc_find2.remove_node(j)
                    
                    mapping=dict()
                    for i in list(G_anc_find2.nodes):
                        mapping[i]=G_anc_find2.nodes[i]['orig_label']
                    G_find2_anc=nx.relabel_nodes(G_anc_find2,mapping)

                    EC_find2_vec[0]=normalised_ec_score(G_find2_anc,G_anc)
                    ICS_find2_vec[0]=ics_score(G_find2_anc,G_anc)
                    S3_find2_vec[0]=s3_score(G_find2_anc,G_anc)
                    network_order_find2[0]=len(G_find2_anc.nodes)

                    #nx.write_edgelist(G_find_int_anc,direc+"anc50_2cherry_branch"+str(branchLength)+"/myalg_int_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                    #nx.write_edgelist(G_true_unint,direc+"anc50_2cherry_branch"+str(branchLength)+"/myalg_unint_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                    #nx.write_edgelist(G_find1_anc,direc+"anc50_2cherry_branch"+str(branchLength)+"/myalgG"+str(l)+"_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                    #nx.write_edgelist(G_find2_anc,direc+"anc50_2cherry_branch"+str(branchLength)+"/myalgG"+str(m)+"_anc_graphs_"+str(l)+"_"+str(m)+".txt")
##########################################
                    G1_orig=copy.deepcopy(G1)
                    G2_orig=copy.deepcopy(G2)

                    t=1
                    tEC=0
               
                    G_anc_find1,G_anc_find2=GFAF_ped_pea(G1_orig,G2_orig,r,q,tolerance=t,toleranceEC=tEC,true_labels=True,divide_by_n=True)
                   
                    print("ancestor found")
                    G_find_int_anc=nx.intersection(G_anc_find1,G_anc_find2)
                    print("my alg edge number int",len(G_find_int_anc.edges()))
                    G_true_unint=graph_intersection_union(G_anc_find1,G_anc_find2)
                    print("my alg edge number unint",len(G_true_unint.edges()))
                    mapping=dict()
                    for i in list(G_find_int_anc.nodes):
                        mapping[i]=G_anc_find1.nodes[i]['orig_label']

                    G_true_unint=nx.relabel_nodes(G_true_unint,mapping)    
                    EC_true_unint_vec[1]=normalised_ec_score(G_true_unint,G_anc)
                    ICS_true_unint_vec[1]=ics_score(G_true_unint,G_anc)
                    S3_true_unint_vec[1]=s3_score(G_true_unint,G_anc)
                    conservedEdges=conserved_edges(G_true_unint,G_anc)
                    conserved_true_unint_vec[1]=conservedEdges
                    extra_edges_true_unint_vec[1]=len(G_true_unint.edges)-conservedEdges
                    missed_edges_true_unint_vec[1]=len(G_anc.edges)-conservedEdges
                    network_order_true_unint[1]=len(G_true_unint.nodes)
                    for j in list(G_find_int_anc.nodes()):
                        if (G_find_int_anc.out_degree(j)==0 and G_find_int_anc.in_degree(j)==0):
                            G_find_int_anc.remove_node(j)
                        elif G_find_int_anc.out_degree(j)==1 and G_find_int_anc.in_degree(j)==1 and (j,j) in list(G_find_int_anc.edges):
                            G_find_int_anc.remove_node(j)
                    
                    
                    
                    G_find_int_anc=nx.relabel_nodes(G_find_int_anc,mapping)

                    EC_find_int_vec[1]=normalised_ec_score(G_find_int_anc,G_anc)
                    ICS_find_int_vec[1]=ics_score(G_find_int_anc,G_anc)
                    S3_find_int_vec[1]=s3_score(G_find_int_anc,G_anc)
                    conservedEdges=conserved_edges(G_find_int_anc,G_anc)
                    conserved_find_int_vec[1]=conservedEdges
                    extra_edges_find_int_vec[1]=len(G_find_int_anc.edges)-conservedEdges
                    missed_edges_find_int_vec[1]=len(G_anc.edges)-conservedEdges
                    network_order_find_int[1]=len(G_find_int_anc.nodes)

                    for j in list(G_anc_find1.nodes()):
                        if (G_anc_find1.out_degree(j)==0 and G_anc_find1.in_degree(j)==0):
                            G_anc_find1.remove_node(j)
                        elif G_anc_find1.out_degree(j)==1 and G_anc_find1.in_degree(j)==1 and (j,j) in list(G_anc_find1.edges):
                            G_anc_find1.remove_node(j)

                    mapping=dict()
                    for i in list(G_anc_find1.nodes):
                        mapping[i]=G_anc_find1.nodes[i]['orig_label']
                    G_find1_anc=nx.relabel_nodes(G_anc_find1,mapping)
                    
                    EC_find1_vec[1]=normalised_ec_score(G_find1_anc,G_anc)
                    ICS_find1_vec[1]=ics_score(G_find1_anc,G_anc)
                    S3_find1_vec[1]=s3_score(G_find1_anc,G_anc)
                    network_order_find1[1]=len(G_find1_anc.nodes)
                    
                    for j in list(G_anc_find2.nodes()):
                        if (G_anc_find2.out_degree(j)==0 and G_anc_find2.in_degree(j)==0):
                            G_anc_find2.remove_node(j)
                        elif G_anc_find2.out_degree(j)==1 and G_anc_find2.in_degree(j)==1 and (j,j) in list(G_anc_find2.edges):
                            G_anc_find2.remove_node(j)
                    
                    mapping=dict()
                    for i in list(G_anc_find2.nodes):
                        mapping[i]=G_anc_find2.nodes[i]['orig_label']
                    G_find2_anc=nx.relabel_nodes(G_anc_find2,mapping)

                    EC_find2_vec[1]=normalised_ec_score(G_find2_anc,G_anc)
                    ICS_find2_vec[1]=ics_score(G_find2_anc,G_anc)
                    S3_find2_vec[1]=s3_score(G_find2_anc,G_anc)
                    network_order_find2[1]=len(G_find2_anc.nodes)

                    #nx.write_edgelist(G_find_int_anc,direc+"anc50_2cherry_branch"+str(branchLength)+"/myalg_int_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                    #nx.write_edgelist(G_true_unint,direc+"anc50_2cherry_branch"+str(branchLength)+"/myalg_unint_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                    #nx.write_edgelist(G_find1_anc,direc+"anc50_2cherry_branch"+str(branchLength)+"/myalgG"+str(l)+"_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                    #nx.write_edgelist(G_find2_anc,direc+"anc50_2cherry_branch"+str(branchLength)+"/myalgG"+str(m)+"_anc_graphs_"+str(l)+"_"+str(m)+".txt")
                  
  pairs_vec=[]
  EC_dmc=[]
  EC_find_int=[]
  EC_true_unint=[]
  S3_dmc=[]
  S3_find_int=[]
  S3_true_unint=[]
  net_order_dmc=[]
  net_order_find1=[]
  net_order_find2=[]
  net_order_find_int=[]
  net_order_true_unint=[]
  conserved_dmc=[]
  conserved_true_unint=[]
  conserved_find_int=[]
  extra_edges_true_unint=[]
  extra_edges_find_int=[]
  missed_edges_true_unint=[]
  missed_edges_find_int=[]

  font = {'family' : 'normal',
          'weight' : 'normal',

          'size'   : 18}


  plt.rc('font', **font)
  pairs_vec=["No divide by n","Divide by n"]
  for i in range(0,2):
              
              if True:
                EC_find_int.append(EC_find_int_vec[i])
                EC_true_unint.append(EC_true_unint_vec[i])

                S3_true_unint.append(S3_true_unint_vec[i])
                S3_find_int.append(S3_find_int_vec[i])

                conserved_true_unint.append(conserved_true_unint_vec[i])
                conserved_find_int.append(conserved_find_int_vec[i])

                
                extra_edges_true_unint.append(extra_edges_true_unint_vec[i])
                extra_edges_find_int.append(extra_edges_find_int_vec[i])

                missed_edges_true_unint.append(missed_edges_true_unint_vec[i])
                missed_edges_find_int.append(missed_edges_find_int_vec[i])

                net_order_find1.append(network_order_find1[i])
                net_order_find2.append(network_order_find2[i])
                net_order_find_int.append(network_order_find_int[i])
                net_order_true_unint.append(network_order_true_unint[i])
              
  print(pairs_vec)
  plt.figure(figsize =(12, 8))
  barWidth = 0.08
  fig = plt.subplots(figsize =(12, 8))
  # Set position of bar on X axis
  br1 = np.arange(len(EC_find_int))
  br2 = [x + barWidth for x in br1]
 
 


  # Make the plot

  plt.bar(br1, EC_find_int, color ='y', width = barWidth,label ="My Algorithm (True Label Int)")
  plt.bar(br2, EC_true_unint, color ='m', width = barWidth,label ="My Algorithm (True Label Unint)")
  

  # Adding Xticks
  plt.xlabel('Leaf Network Pair')
  plt.ylabel('EC Score of Predicted and Actual Ancestor')
  plt.xticks([r + barWidth+0.16 for r in range(len(EC_dmc))],
          pairs_vec)

  # Put a legend to the right of the current axis
  plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

  plt.savefig(direc+"anc50_2cherry_branch"+str(branchLength)+"/EC_scores_branchlength"+str(branchLength)+".png",bbox_inches="tight")

  plt.figure(figsize =(12, 8))
  barWidth = 0.08
  fig = plt.subplots(figsize =(12, 8))
  # Set position of bar on X axis
  br1 = np.arange(len(EC_find_int))
  br2 = [x + barWidth for x in br1]




  # Make the plot

  plt.bar(br1, S3_find_int, color ='y', width = barWidth,label ="My Algorithm (True Label Int)")
  plt.bar(br2, S3_true_unint, color ='m', width = barWidth,label ="My Algorithm (True Label Unint)")
  

  # Adding Xticks
  plt.xlabel('Leaf Network Pair')
  plt.ylabel('S3 Score of Predicted and Actual Ancestor')
  plt.xticks([r + barWidth+0.16 for r in range(len(S3_dmc))],
          pairs_vec)

  plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
  plt.savefig(direc+"anc50_2cherry_branch"+str(branchLength)+"/S3_scores_branchlength"+str(branchLength)+".png",bbox_inches='tight')

  plt.figure(figsize =(12, 8))
  barWidth = 0.08
  fig = plt.subplots(figsize =(12, 8))
  # Set position of bar on X axis
  br1 = np.arange(len(EC_find_int))
  br2 = [x + barWidth for x in br1]
  br3 = [x + barWidth for x in br2]
  br4 = [x + barWidth for x in br3]




  # Make the plot


  plt.bar(br1, net_order_find_int, color ='y', width = barWidth,label ="My Algorithm (True Label Int)")
  plt.bar(br2, net_order_true_unint, color ='m', width = barWidth,label ="My Algorithm (True Label Unint)")
  plt.bar(br3, net_order_find1, color ='grey', width = barWidth,label ="My Algorithm G1")
  plt.bar(br4, net_order_find2, color ='gold', width = barWidth,label ="My Algorithm G2")


  # Adding Xticks
  plt.xlabel('Leaf Network Pair')
  plt.ylabel('Network Order of Predicted and Actual Ancestor')
  plt.xticks([r + barWidth+0.16 for r in range(len(net_order_dmc))],
          pairs_vec)

  plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
  plt.savefig(direc+"anc50_2cherry_branch"+str(branchLength)+"/net_order_scores_branchlength"+str(branchLength)+".png",bbox_inches='tight')

  plt.figure(figsize =(12, 8))
  barWidth = 0.08
  fig = plt.subplots(figsize =(12, 8))
  # Set position of bar on X axis
  br1 = np.arange(len(conserved_find_int))
  br2 = [x + barWidth for x in br1]



  # Make the plot


  plt.bar(br1, conserved_find_int, color ='y', width = barWidth,label ="My Algorithm (True Label Int)")
  plt.bar(br2, conserved_true_unint, color ='m', width = barWidth,label ="My Algorithm (True Label Unint)")


  # Adding Xticks
  plt.xlabel('Leaf Network Pair')
  plt.ylabel('No. of Correct Edges in Predicted Ancestor')
  plt.xticks([r + barWidth+0.16 for r in range(len(conserved_dmc))],
          pairs_vec)

  # Put a legend to the right of the current axis
  plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

  plt.savefig(direc+"anc50_2cherry_branch"+str(branchLength)+"/conserved_edges_branchlength"+str(branchLength)+".png",bbox_inches="tight")


  plt.figure(figsize =(12, 8))

  barWidth = 0.08
  fig = plt.subplots(figsize =(12, 8))
  # Set position of bar on X axis
  br1 = np.arange(len(conserved_find_int))
  br2 = [x + barWidth for x in br1]



  # Make the plot


  plt.bar(br1, extra_edges_find_int, color ='y', width = barWidth,label ="My Algorithm (True Label Int)")
  plt.bar(br2, extra_edges_true_unint, color ='m', width = barWidth,label ="My Algorithm (True Label Unint)")


  # Adding Xticks
  plt.xlabel('Leaf Network Pair')
  plt.ylabel('No. of Extra Edges in Predicted Ancestor')
  plt.xticks([r + barWidth+0.16 for r in range(len(conserved_dmc))],
          pairs_vec)

  # Put a legend to the right of the current axis
  plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

  plt.savefig(direc+"anc50_2cherry_branch"+str(branchLength)+"/extra_edges_branchlength"+str(branchLength)+".png",bbox_inches="tight")


  plt.figure(figsize =(12, 8))

  barWidth = 0.08
  fig = plt.subplots(figsize =(12, 8))
  # Set position of bar on X axis
  br1 = np.arange(len(conserved_find_int))
  br2 = [x + barWidth for x in br1]



  # Make the plot

  plt.bar(br1, missed_edges_find_int, color ='y', width = barWidth,label ="My Algorithm (True Label Int)")
  plt.bar(br2, missed_edges_true_unint, color ='m', width = barWidth,label ="My Algorithm (True Label Unint)")


  # Adding Xticks
  plt.xlabel('Leaf Network Pair')
  plt.ylabel('No. Edges in True Ancestor Missed By Predicted Ancestor')
  plt.xticks([r + barWidth+0.16 for r in range(len(conserved_dmc))],
          pairs_vec)

  # Put a legend to the right of the current axis
  plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

  plt.savefig(direc+"anc50_2cherry_branch"+str(branchLength)+"/missed_edges_branchlength"+str(branchLength)+".png",bbox_inches="tight")
  write_list_to_file([algType],"test_datasets_ancestral/regulatory_test_ped_pea/anc50_2cherry_branch"+str(branchLength)+"/recent_run")

#def varying_r_ancestral(q=0.4,direc="test_datasets_ancestral/regulatory_test_ped_pea/"):

#def varying_q_ancestral(r=1,direc="test_datasets_ancestral/regulatory_test_ped_pea/"):

#def with_or_without_scoring_ancestral(r=1,q=0.4,direc="test_datasets_ancestral/regulatory_test_ped_pea/"):