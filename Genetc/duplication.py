
#from re import M, S
from os import dup
from time import time
from matplotlib.pyplot import xcorr
import networkx as nx
import itertools
import copy
from networkx.readwrite.json_graph import tree
import numpy as np
from networkx.algorithms.shortest_paths.unweighted import all_pairs_shortest_path, predecessor
import math
from collections import defaultdict
from networkx.utils import py_random_state
import random as random
import json
import matplotlib.pyplot as plt
from scipy.stats import expon as ex 
#from utilities import connected_component_subgraphs
from Genetc.utilities import *
import random as rd
import igraph as ig
from Genetc import utilities as ut
#------------------------------------------------------
#Network evolution models

def duplicate_genes_igraph(G,genes,iteration=0,self_loops=True):
  edgeAddVec=[]
  n=G.vcount()
  for i in genes:

    G.add_vertices([n])
    G.vs[n]['name']=str(G.vs[i]["name"])+"_"+str(iteration)

  for j in genes:
      
    for i in G.neighbors(j,mode="out"):
      if i!=j:  
        edgeAddVec.append((n,i))
      
    for i in G.neighbors(j,mode="in"):
      if i!=j:
        edgeAddVec.append((i,n))
        
    if self_loops:
      if j in G.neighbors(j,mode="all"):
        edgeAddVec.append((n,n))
        edgeAddVec.append((j,n))
        edgeAddVec.append((n,j))
  G.add_edges(edgeAddVec)
  return G

def PED_PEA_igraph(G_n,r,q,iteration=0,copyG=False,duplicate_mutation_rate=0.5,edge_add_type='standard',edge_add_TF_only=True,isolates=False):
  ###print(iteration)
  edgeDelList=[]
  if copyG==True:
    G=copy.deepcopy(G_n)
  else:
    G=G_n
  
  n=G.vcount()
  
  dupNode= int(n * random.random())
  dupDegree=G.degree(dupNode)

  G=duplicate_genes_igraph(G,[dupNode],iteration=iteration)
  copyNode=G.vs.find(name=str(G.vs[dupNode]['name'])+"_"+str(iteration)).index

  parents= G.neighbors(dupNode,mode="in")
  children=G.neighbors(dupNode,mode="out")
  #print(parents,children)
  parentsCopy=set(parents)
  childrenCopy=set(children)
  nodeList=[i for i in range(G.vcount())]

  parentListRemoved=set(list(nodeList))-parentsCopy
  childListRemoved=set(list(nodeList))-childrenCopy

  theRemovalists=childrenCopy.union(set(parentsCopy))
  #nodeListRemoved=set(list(nodeList))-theRemovalists
  #print(parents,children)
  if q>0:
    
    if dupNode in G.neighbors(dupNode,mode="all"):
        rando=random.random()  
        if rando <= q:
          rando=random.random()
            
          if rando >0.5:
            edgeDelList.append((dupNode,dupNode))
            #print("delete",dupNode,dupNode)
          else:
            edgeDelList.append((copyNode,copyNode))
            #print("delete",copyNode,copyNode)
    if dupNode in G.neighbors(copyNode,mode="out"):
        rando=random.random()
        if rando <= q:
          rando=random.random()
          if rando >0.5:
            edgeDelList.append((dupNode,copyNode))
            #print("delete",dupNode,copyNode)
          else:
            edgeDelList.append((copyNode,dupNode))
            #print("delete",copyNode,dupNode)
    for i in parents:
      if i!=copyNode and i!= dupNode:
        rando=random.random()
        if rando <= q:
          rando=random.random()
          #print("i is",i,"copyNode is",copyNode,"dupNode is",dupNode)
          if rando >1-duplicate_mutation_rate:
            edgeDelList.append((i,copyNode))
            #print("delete",i,copyNode)
          else:
            edgeDelList.append((i,dupNode))
            #print("delete",i,dupNode)
    #children=copy.deepcopy(list(G.successors(dupNode)))
    children=list(G.successors(dupNode))
    #print(children)
    for i in children:
      
      if i!= copyNode and i!= dupNode:
        rando=random.random()
        
        if rando <= q:
          rando=random.random()
          #print("i is",i,"copyNode is",copyNode,"dupNode is",dupNode)
          if rando >1-duplicate_mutation_rate:
            edgeDelList.append((copyNode,i))
            #print("delete",copyNode,i)
          else:
            edgeDelList.append((dupNode,i))
            #print("delete",dupNode,i)
  G.delete_edges(edgeDelList)
  if r>0:
    if edge_add_TF_only==False:
      if edge_add_type=='standard':
        for i in childListRemoved:
          
          
          if i==dupNode:
                rando = random.random()
              
                if rando<r/n:
                  rando = random.random()
                  
                  if rando>0.5:
                        G.add_edges([(dupNode,dupNode)])
                  else:
                        G.add_edges([(copyNode,copyNode)])
          elif i==copyNode:
                rando = random.random()
              
                if rando<r/n:
                  rando = random.random()
                  if rando>0.5:
                        G.add_edges([(copyNode,dupNode)])
                  else:
                        G.add_edges([(dupNode,copyNode)])
            
          else:  
                  rando = random.random()
                  if rando<r/n:
                    
                    rando=random.random()

                    if rando>0.5:
                      G.add_edges([(dupNode,i)])
                    else:
                      G.add_edges([(copyNode,i)])

        for i in parentListRemoved:
          
          if i!=dupNode and i!=copyNode:
            
              rando = random.random()
              if rando<r/n:
                
                  rando=random.random()
                  if rando>0.5:
                    G.add_edges([(i,dupNode)])
                  else:
                    G.add_edges([(i,copyNode)])


      elif edge_add_type=='uniform':
        
        for i in childListRemoved:
          
            
            if i==dupNode:  
                  rando = random.random()
                
                  if rando<r/(2*n-dupDegree+1):
                    rando = random.random()

                    if rando>0.5:
                        G.add_edges([(dupNode,dupNode)])
                    else:
                        G.add_edges([(copyNode,copyNode)])
            elif i==copyNode:    
                  rando = random.random()
                
                  if rando<r/(2*n-dupDegree+1):
                    rando = random.random()
                    if rando>0.5:
                        G.add_edges([(copyNode,dupNode)])
                    else:
                        G.add_edges([(dupNode,copyNode)])
              
            else:  
              rando = random.random()
              if rando<r/(2*n-dupDegree+1):
                rando=random.random()

                if rando>0.5:
                      G.add_edges([(dupNode,i)])
                else:
                      G.add_edges([(copyNode,i)])

        for i in parentListRemoved:
          
          if i!=dupNode and i!=copyNode:
            
              rando = random.random()
              if rando<r/(2*n-dupDegree+1):
                
                  rando=random.random()
                  if rando>0.5:
                    G.add_edges([(i,dupNode)])
                  else:
                    G.add_edges([(i,copyNode)])

      elif edge_add_type=='preferential':
        
        for i in childListRemoved:
          
            if i ==dupNode:
              
                  rando = random.random()
                
                  if rando<r*dupDegree/(2*n-dupDegree+1):
                    rando = random.random()

                    if rando>0.5:
                        G.add_edges([(dupNode,dupNode)])
                    else:
                        G.add_edges([(copyNode,copyNode)])
                
                
            elif i==copyNode:
                  rando = random.random()
                  if rando<r*dupDegree/(2*n-dupDegree+1):
                    rando = random.random()
                    if rando>0.5:
                        G.add_edges([(copyNode,dupNode)])
                    else:
                        G.add_edges([(dupNode,copyNode)])
              
            else:  
              rando = random.random()
              if rando<r*dupDegree/(2*n-dupDegree+1):
                rando=random.random()

                if rando>0.5:
                      G.add_edges([(dupNode,i)])
                else:
                      G.add_edges([(copyNode,i)])

        for i in parentListRemoved:
          
          if i!=dupNode and i!=copyNode:
            
              rando = random.random()
              if rando<r*dupDegree/(2*n-dupDegree+1):
              
                  rando=random.random()
                  if rando>0.5:
                    G.add_edges([(i,dupNode)])
                  else:
                    G.add_edges([(i,copyNode)])

      else:
        raise TypeError("Edge addition type not implemented. Please choose from standard, unfirom, or preferential")
    else:
      if edge_add_type=='standard':
        if G.outdegree(dupNode)!=0 and G.outdegree(copyNode)!=0:
          for i in childListRemoved:
              if i==dupNode:
                    rando = random.random()
                  
                    if rando<r/n:
                      rando = random.random()

                      if rando>0.5:
                        G.add_edges([(dupNode,dupNode)])
                      else:
                        G.add_edges([(copyNode,copyNode)])
                  
                  
              elif i==copyNode:
                    rando = random.random()
                    if rando<r/n:
                      rando = random.random()
                      if rando>0.5:
                        G.add_edges([(copyNode,dupNode)])
                      else:
                        G.add_edges([(dupNode,copyNode)])
                
              else:  
                  rando = random.random()
                  if rando<r/n:
                    
                    rando=random.random()

                    if rando>0.5:
                      G.add_edges([(dupNode,i)])
                    else:
                      G.add_edges([(copyNode,i)])

        for i in parentListRemoved:
          
          if i!=dupNode and i!=copyNode:
            
              rando = random.random()
              if rando<r/n:
                if G.outdegree(i)!=0:  
                  rando=random.random()
                  if rando>0.5:
                    G.add_edges([(i,dupNode)])
                  else:
                    G.add_edges([(i,copyNode)])

      elif edge_add_type=='uniform':
        if G.outdegree(dupNode)!=0 and G.outdegree(copyNode)!=0:
          for i in childListRemoved:
          
            if i ==dupNode:
              
                  rando = random.random()
                
                  if rando<r/(2*n-dupDegree+1):
                    rando = random.random()

                    if rando>0.5:
                        G.add_edges([(dupNode,dupNode)])
                    else:
                        G.add_edges([(copyNode,copyNode)])
                
                
            elif i==copyNode:
                  rando = random.random()
                  if rando<r/(2*n-dupDegree+1):
                    rando = random.random()
                    if rando>0.5:
                        G.add_edges([(copyNode,dupNode)])
                    else:
                        G.add_edges([(dupNode,copyNode)])
              
            else:  
              rando = random.random()
              if rando<r/(2*n-dupDegree+1):
                rando=random.random()

                if rando>0.5:
                      G.add_edges([(dupNode,i)])
                else:
                      G.add_edges([(copyNode,i)])

        for i in parentListRemoved:
          
          if i!=dupNode and i!=copyNode:
            
              rando = random.random()
              if rando<r/(2*n-dupDegree+1):
                if G.outdegree(i)!=0:  
                  rando=random.random()
                  if rando>0.5:
                    G.add_edges([(i,dupNode)])
                  else:
                    G.add_edges([(i,copyNode)])

      elif edge_add_type=='preferential':
        if G.outdegree(dupNode)!=0 and G.outdegree(copyNode)!=0:
          for i in childListRemoved:
          
            if i ==dupNode:
              
                  rando = random.random()
                
                  if rando<r*dupDegree/(2*n-dupDegree+1):
                    rando = random.random()

                    if rando>0.5:
                        G.add_edges([(dupNode,dupNode)])
                    else:
                        G.add_edges([(copyNode,copyNode)])
                
                
            elif i==copyNode:
                  rando = random.random()
                  if rando<r*dupDegree/(2*n-dupDegree+1):
                    rando = random.random()
                    if rando>0.5:
                        G.add_edges([(copyNode,dupNode)])
                    else:
                        G.add_edges([(dupNode,copyNode)])
              
            else:  
              rando = random.random()
              if rando<r*dupDegree/(2*n-dupDegree+1):
                rando=random.random()

                if rando>0.5:
                      G.add_edges([(dupNode,i)])
                else:
                      G.add_edges([(copyNode,i)])

        for i in parentListRemoved:
          
          if i!=dupNode and i!=copyNode:
            
              rando = random.random()
              if rando<r*dupDegree/(2*n-dupDegree+1):
                if G.outdegree(i)!=0:  
                  rando=random.random()
                  if rando>0.5:
                    G.add_edges([(i,dupNode)])
                  else:
                    G.add_edges([(i,copyNode)])

      else:
        raise TypeError("Edge addition type not implemented. Please choose from standard, unfirom, or preferential")
  if not isolates:
    if G.degree(dupNode)==0:
      G.delete_vertices([dupNode])
    else:
      
      if G.degree(copyNode)==0:
        G.delete_vertices([copyNode])  
  return G

def PED_PEA_looper(G,r,q,steps,copyG=False,duplicate_mutation_rate=0.5,edge_add_type='standard',edge_add_TF_only=True,isolates=False):
  m=0
  while G.vcount()<steps:
    
    m=m+1
    G=PED_PEA_igraph(G,r,q,iteration=m,copyG=copyG,duplicate_mutation_rate=duplicate_mutation_rate,edge_add_type=edge_add_type,edge_add_TF_only=edge_add_TF_only,isolates=isolates)

  return G

#Undirected versions

def duplicate_genes_igraph_und(G,genes,iteration=0,self_loops=True):
  edgeAddVec=[]
  n=G.vcount()
  for i in genes:

    G.add_vertices([n])
    G.vs[n]['name']=str(G.vs[i]["name"])+"_"+str(iteration)

  for j in genes:
      
    for i in G.neighbors(j):
      if i!=j:  
        edgeAddVec.append((n,i))
    
  G.add_edges(edgeAddVec)
  return G

def PED_PEA_igraph_und(G_n,r,q,iteration=0,copyG=False,duplicate_mutation_rate=0.5,edge_add_type='standard',isolates=False,model='PED',keep_labels=False):
  ###print(iteration)
  edgeDelList=[]
  if copyG==True:
    G=copy.deepcopy(G_n)
  else:
    G=G_n
  
  n=G.vcount()
  
  dupNode= int(n * random.random())
  dupDegree=G.degree(dupNode)

  G=duplicate_genes_igraph_und(G,[dupNode],iteration=iteration)
  copyNode=G.vs.find(name=str(G.vs[dupNode]['name'])+"_"+str(iteration)).index
  #print("copyNode is",copyNode,"dupNode is",dupNode)
  #print("node dupd is",str(G.vs[dupNode]['name']))
  neighs= G.neighbors(dupNode)
  
  #print(parents,children)
  neighsCopy=set(neighs)
  
  nodeList=[i for i in range(G.vcount())]

  neighListRemoved=set(list(nodeList))-neighsCopy
  

  theRemovalists=neighsCopy.union(set(neighsCopy))
  #nodeListRemoved=set(list(nodeList))-theRemovalists
  #print(parents,children)
  if model=='PED':
    if q>0:
      
      
      for i in neighs:
        
          rando=random.random()
          if rando <= q:
            rando=random.random()
            #print("i is",i,"copyNode is",copyNode,"dupNode is",dupNode)
            if rando >0.5:
              edgeDelList.append((i,copyNode))
              #print("delete",i,copyNode)
            else:
              edgeDelList.append((dupNode,i))
              #print("delete",i,dupNode)
      
    
    G.delete_edges(edgeDelList)
    if r>0:
        #if edge_add_TF_only==False:
        if edge_add_type=='standard':
          for i in neighListRemoved:
                    rando = random.random()
                    if rando<r/n:
                      
                        rando=random.random()

                        if rando>0.5:
                          if i!=dupNode:
                            G.add_edges([(dupNode,i)])
                        else:
                          if i!=copyNode:
                            G.add_edges([(copyNode,i)])

        elif edge_add_type=='uniform':
          
          for i in neighListRemoved:
            
        
                rando = random.random()
                if rando<r/(2*n-dupDegree+1):
                  rando=random.random()

                  if rando>0.5:
                        if i!=dupNode:
                            G.add_edges([(dupNode,i)])
                        else:
                          if i!=copyNode:
                            G.add_edges([(copyNode,i)])

        elif edge_add_type=='preferential':
          
          for i in neighListRemoved:
          
                rando = random.random()
                if rando<r*dupDegree/(2*n-dupDegree+1):
                  rando=random.random()

                  if rando>0.5:
                        if i!=dupNode:
                            G.add_edges([(dupNode,i)])
                        else:
                          if i!=copyNode:
                            G.add_edges([(copyNode,i)])

        else:
          raise TypeError("Edge addition type not implemented. Please choose from standard, unfirom, or preferential")
  if model=='sole':
    if q>0:
      for i in neighs:
        rando=random.random()
        if rando <= q:
          edgeDelList.append((i,copyNode))
          
    
    #print([(i.source,i.target) for i in G.es])
    #print(edgeDelList)
    G.delete_edges(edgeDelList)
    if r>0:
        #if edge_add_TF_only==False:
        if edge_add_type=='standard':
          for i in neighListRemoved:
                    rando = random.random()
                    if rando<r/n:
                      if i!=copyNode:
                        G.add_edges([(copyNode,i)])
                        #print("add",(copyNode,i))
        elif edge_add_type=='uniform':
          
          for i in neighListRemoved:
            
        
                rando = random.random()
                if rando<r/(2*n-dupDegree+1):
                  if i!=copyNode:
                    G.add_edges([(copyNode,i)])

        elif edge_add_type=='preferential':
          
          for i in neighListRemoved:
          
                rando = random.random()
                if rando<r*dupDegree/(2*n-dupDegree+1):
                  if i!=copyNode:
                    G.add_edges([(copyNode,i)])

        else:
          raise TypeError("Edge addition type not implemented. Please choose from standard, unfirom, or preferential")
    
  if not isolates:
    if G.degree(dupNode)==0:
      G.delete_vertices([dupNode])
    else:
      
      if G.degree(copyNode)==0:
        G.delete_vertices([copyNode])  
  return G

def PED_PEA_looper_und(G,r,q,steps,copyG=False,duplicate_mutation_rate=0.5,edge_add_type='standard',isolates=False,model='PED',keep_labels=False):
  m=0
  while G.vcount()<steps:
    
    m=m+1
    #print(m)
    G=PED_PEA_igraph_und(G,r,q,iteration=m,copyG=copyG,duplicate_mutation_rate=duplicate_mutation_rate,edge_add_type=edge_add_type,isolates=isolates,model=model,keep_labels=keep_labels)

  return G

#directed but no self loops

def duplicate_genes_igraph_nsl(G,genes,iteration=0,self_loops=True,labels=False):
  edgeAddVec=[]
  n=G.vcount()
  for i in genes:
    #print(n)
    G.add_vertices([n])
    G.vs[n]['name']=str(G.vs[i]["name"])+"_"+str(iteration)
    if labels:
      try:
        G.vs[n]['family']=G.vs[i]['family']
      except:
         print("No family name defined for",i)
      try:
        G.vs[n]['ancestor']=G.vs[i]['ancestor']
      except:
        print("No ancestor name defined for",i)
      

  for j in genes:
      
    for i in G.neighbors(j,mode="out"):
      if i!=j:  
        edgeAddVec.append((n,i))
      
    for i in G.neighbors(j,mode="in"):
      if i!=j:
        edgeAddVec.append((i,n))
        
  G.add_edges(edgeAddVec)
  return G

def PED_PEA_igraph_nsl(G_n,r,q,iteration=0,copyG=False,duplicate_mutation_rate=0.5,edge_add_type='standard',edge_add_TF_only=True,isolates=False,model='PED',labels=False):
  ###print(iteration)
  edgeDelList=[]
  if copyG==True:
    G=copy.deepcopy(G_n)
  else:
    G=G_n
  
  n=G.vcount()
  
  dupNode= int(n * random.random())
  dupDegree=G.degree(dupNode)
  #print(iteration,'iteration')
  G=duplicate_genes_igraph_nsl(G,[dupNode],iteration=iteration,labels=labels)
  #print("copynode true ID",G.vs[n])
  copyNode=G.vs.find(name=str(G.vs[dupNode]['name'])+"_"+str(iteration)).index

  parents= G.neighbors(dupNode,mode="in")
  children=G.neighbors(dupNode,mode="out")
  #print(parents,children)
  parentsCopy=set(parents)
  childrenCopy=set(children)
  nodeList=[i for i in range(G.vcount())]

  parentListRemoved=set(list(nodeList))-parentsCopy
  childListRemoved=set(list(nodeList))-childrenCopy

  theRemovalists=childrenCopy.union(set(parentsCopy))
  #nodeListRemoved=set(list(nodeList))-theRemovalists
  #print("parents",parents)
  if q>0:
    for i in parents:
      if i!=copyNode and i!= dupNode:
        rando=random.random()
        if rando <= q:
          if model=='PED':
            rando=random.random()
            
            if rando >1-duplicate_mutation_rate:
              edgeDelList.append((i,copyNode))
              
            else:
              edgeDelList.append((i,dupNode))
             
          elif model=='sole':
            
            edgeDelList.append((i,copyNode))
              
    children=list(G.successors(dupNode))
    #print("children",children)
    for i in children:
      
      if i!= copyNode and i!= dupNode:
        rando=random.random()
        
        if rando <= q:
          if model=='PED':
            rando=random.random()
            #print("i is",i,"copyNode is",copyNode,"dupNode is",dupNode)
            if rando >1-duplicate_mutation_rate:
              edgeDelList.append((copyNode,i))
              #print("delete",copyNode,i)
            else:
              edgeDelList.append((dupNode,i))
              #print("delete",dupNode,i)
          elif model=='sole':
            
            edgeDelList.append((copyNode,i))
  #print("dupnode",dupNode,"copynode",copyNode)
  #print(G.vs[dupNode],G.vs[copyNode])
  #print([(e.source,e.target) for e in G.es()])
  #print(edgeDelList)
  G.delete_edges(edgeDelList)
  if r>0:
    if edge_add_TF_only==False:
      if edge_add_type=='standard':
        for i in childListRemoved:
          if i==copyNode:
                rando = random.random()
                if rando<r/n:
                  if model=='PED':
                    rando = random.random()
                    if rando>0.5:
                          G.add_edges([(copyNode,dupNode)])
                    else:
                          G.add_edges([(dupNode,copyNode)])
                  elif model=='sole':
                    G.add_edges([(copyNode,dupNode)])
              
          elif i!=dupNode and i!=copyNode:  
                  rando = random.random()
                  if rando<r/n:
                    if model=='PED':
                      rando=random.random()

                      if rando>0.5:
                        G.add_edges([(dupNode,i)])
                      else:
                        G.add_edges([(copyNode,i)])
                    elif model=='sole':
                      G.add_edges([(copyNode,i)])

        for i in parentListRemoved:
          
          if i!=dupNode and i!=copyNode:
            
              rando = random.random()
              if rando<r/n:
                if model=='PED':
                  rando=random.random()
                  if rando>0.5:
                    G.add_edges([(i,dupNode)])
                  else:
                    G.add_edges([(i,copyNode)])
                elif model=='sole':
                  G.add_edges([(i,copyNode)])


      elif edge_add_type=='uniform':
        
        for i in childListRemoved:
          
            
            if i==copyNode:  
                
                  rando = random.random()
                
                  if rando<r/(2*n-dupDegree+1):
                    if model=='PED':  
                      rando = random.random()
                      if rando>0.5:
                          G.add_edges([(copyNode,dupNode)])
                      else:
                          G.add_edges([(dupNode,copyNode)])
                    elif model=='sole':
                      G.add_edges([(copyNode,dupNode)])
              
            elif i!=dupNode and i!=copyNode:  
             
              rando = random.random()
              if rando<r/(2*n-dupDegree+1):
                if model=='PED':
                  rando=random.random()

                  if rando>0.5:
                        G.add_edges([(dupNode,i)])
                  else:
                        G.add_edges([(copyNode,i)])
                elif model=='sole':
                  G.add_edges([(copyNode,i)])

        for i in parentListRemoved:
          
          if i!=dupNode and i!=copyNode:
            
              rando = random.random()
              if rando<r/(2*n-dupDegree+1):
                if model=='PED':
                  rando=random.random()
                  if rando>0.5:
                    G.add_edges([(i,dupNode)])
                  else:
                    G.add_edges([(i,copyNode)])
                elif model=='sole':
                  G.add_edges([(i,copyNode)])

      elif edge_add_type=='preferential':
        
        for i in childListRemoved:
          
                
            if i==copyNode:
                  rando = random.random()
                  if rando<r*dupDegree/(2*n-dupDegree+1):
                    if model=='PED':
                      rando = random.random()
                      if rando>0.5:
                          G.add_edges([(copyNode,dupNode)])
                      else:
                          G.add_edges([(dupNode,copyNode)])
                    elif model=='sole':
                      G.add_edges([(copyNode,dupNode)])
              
            elif i!=dupNode and i!=copyNode:  
              rando = random.random()
              if rando<r*dupDegree/(2*n-dupDegree+1):
                if model=='PED':
                  rando=random.random()

                  if rando>0.5:
                        G.add_edges([(dupNode,i)])
                  else:
                        G.add_edges([(copyNode,i)])
                elif model=='sole':
                  G.add_edges([(copyNode,i)])
        for i in parentListRemoved:
          
          if i!=dupNode and i!=copyNode:
            
              rando = random.random()
              if rando<r*dupDegree/(2*n-dupDegree+1):
                if model=='PED':
              
                  rando=random.random()
                  if rando>0.5:
                    G.add_edges([(i,dupNode)])
                  else:
                    G.add_edges([(i,copyNode)])
                elif model=='sole':   
                  G.add_edges([(i,copyNode)])

      else:
        raise TypeError("Edge addition type not implemented. Please choose from standard, unfirom, or preferential")
    else:
      if edge_add_type=='standard':
        if G.outdegree(dupNode)!=0 and G.outdegree(copyNode)!=0:
          for i in childListRemoved:
              
                  
              if i==copyNode:
                    rando = random.random()
                    if rando<r/n:
                      if model=='PED':
                        rando = random.random()
                        if rando>0.5:
                          G.add_edges([(copyNode,dupNode)])
                        else:
                          G.add_edges([(dupNode,copyNode)])
                      elif model=='sole':
                        G.add_edges([(copyNode,dupNode)])
                
              elif i!=dupNode and i!=copyNode:  
                  rando = random.random()
                  if rando<r/n:
                    if model=='PED': 
                      rando=random.random()

                      if rando>0.5:
                        G.add_edges([(dupNode,i)])
                      else:
                        G.add_edges([(copyNode,i)])
                    elif model=='sole':
                      G.add_edges([(copyNode,i)])

        for i in parentListRemoved:
          
          if i!=dupNode and i!=copyNode:
            
              rando = random.random()
              if rando<r/n:
                if G.outdegree(i)!=0:  
                  if model=='PED':
                    rando=random.random()
                    if rando>0.5:
                      G.add_edges([(i,dupNode)])
                    else:
                      G.add_edges([(i,copyNode)])
                  elif model=='sole':
                    G.add_edges([(i,copyNode)])

      elif edge_add_type=='uniform':
        if G.outdegree(dupNode)!=0 and G.outdegree(copyNode)!=0:
          for i in childListRemoved:
        
            if i==copyNode:
                  rando = random.random()
                  if rando<r/(2*n-dupDegree+1):
                    if model=='PED':
                      rando = random.random()
                      if rando>0.5:
                          G.add_edges([(copyNode,dupNode)])
                      else:
                          G.add_edges([(dupNode,copyNode)])
                    elif model=='sole':
                      G.add_edges([(copyNode,dupNode)])
              
            elif i!=dupNode and i!=copyNode:  
              rando = random.random()
              if rando<r/(2*n-dupDegree+1):
                if model=='PED':
                  rando=random.random()

                  if rando>0.5:
                        G.add_edges([(dupNode,i)])
                  else:
                        G.add_edges([(copyNode,i)])
                elif model=='sole':
                  G.add_edges([(copyNode,i)])

        for i in parentListRemoved:
          
          if i!=dupNode and i!=copyNode:
            
              rando = random.random()
              if rando<r/(2*n-dupDegree+1):
                if G.outdegree(i)!=0:  
                  if model=='PED':
                    rando=random.random()
                    if rando>0.5:
                      G.add_edges([(i,dupNode)])
                    else:
                      G.add_edges([(i,copyNode)])
                  elif model=='sole':
                    G.add_edges([(i,copyNode)])

      elif edge_add_type=='preferential':
        if G.outdegree(dupNode)!=0 and G.outdegree(copyNode)!=0:
          for i in childListRemoved:
          
                
                
            if i==copyNode:
                  rando = random.random()
                  if rando<r*dupDegree/(2*n-dupDegree+1):
                    if model=='PED':
                      rando = random.random()
                      if rando>0.5:
                          G.add_edges([(copyNode,dupNode)])
                      else:
                          G.add_edges([(dupNode,copyNode)])
                    elif model=='sole':
                      G.add_edges([(copyNode,dupNode)])
              
            elif i!=dupNode and i!=copyNode:  
              rando = random.random()
              if rando<r*dupDegree/(2*n-dupDegree+1):
                if model=='PED':
                  rando=random.random()

                  if rando>0.5:
                        G.add_edges([(dupNode,i)])
                  else:
                        G.add_edges([(copyNode,i)])
                elif model=='sole':
                  G.add_edges([(copyNode,i)])

        for i in parentListRemoved:
          
          if i!=dupNode and i!=copyNode:
            
              rando = random.random()
              if rando<r*dupDegree/(2*n-dupDegree+1):
                if G.outdegree(i)!=0:  
                  if model=='PED':
                    rando=random.random()
                    if rando>0.5:
                      G.add_edges([(i,dupNode)])
                    else:
                      G.add_edges([(i,copyNode)])

                  elif model=='sole':   
                    G.add_edges([(i,copyNode)])
      else:
        raise TypeError("Edge addition type not implemented. Please choose from standard, unfirom, or preferential")
  if not isolates:
    if G.degree(dupNode,mode='all')==0:
      G.delete_vertices([dupNode])
      #print("deleted",dupNode)
    else:
      
      if G.degree(copyNode,mode='all')==0:
        G.delete_vertices([copyNode])
        #print("deleted",copyNode)  
  return G

def PED_PEA_looper_nsl(G,r,q,steps,copyG=False,duplicate_mutation_rate=0.5,edge_add_type='standard',edge_add_TF_only=True,isolates=False,isolateMethod='progressive',model='PED',labels=False,iteration=0):
  m=iteration
  if not isolates and isolateMethod=='progressive':
    while G.vcount()<steps:
      
      m=m+1
      
      G=PED_PEA_igraph_nsl(G,r,q,iteration=m,copyG=copyG,duplicate_mutation_rate=duplicate_mutation_rate,edge_add_type=edge_add_type,edge_add_TF_only=edge_add_TF_only,isolates=isolates,model=model,labels=labels)
      #print(ut.proportion_of_nodes_that_have_self_loops(G))
  elif not isolates and isolateMethod=='final':
    while G.vcount()<steps:
      m=m+1
      
      G=PED_PEA_igraph_nsl(G,r,q,iteration=m,copyG=copyG,duplicate_mutation_rate=duplicate_mutation_rate,edge_add_type=edge_add_type,edge_add_TF_only=edge_add_TF_only,isolates=False,model=model,labels=labels)
      
    for i in G.vs():
      if G.degree(i,mode='all')==0:
        G.delete_vertices([i])
  else:
     while G.vcount()<steps:
      
      m=m+1
      
      G=PED_PEA_igraph_nsl(G,r,q,iteration=m,copyG=copyG,duplicate_mutation_rate=duplicate_mutation_rate,edge_add_type=edge_add_type,edge_add_TF_only=edge_add_TF_only,isolates=isolates,model=model,labels=labels)
      #print(ut.proportion_of_nodes_that_have_self_loops(G))
   
  return G
