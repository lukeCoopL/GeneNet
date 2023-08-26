#from re import M, S
from time import time
from xml.dom.minicompat import NodeList
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
from Genetc.alignment import *
from Genetc.duplication import *
import igraph as ig
#Utilities
#----------------------------------------------
def print_var_name(variable):
  for theName in globals():
    if eval(theName) == variable:
      return theName

def read_list_as_float(strang):
  vecc=[]
  with open(strang) as f:
    lines = f.read().splitlines()
  for i in lines:
    vecc.append(float(i))
  
  return vecc

def write_list_to_file(thelist,name):
  textfile = open(str(name)+".txt", "w")
  for element in thelist:
      textfile.write(str(element) + "\n")
  textfile.close()

def write_dict_to_file(theDict,name):

  with open(name, 'w') as convert_file:
    convert_file.write(json.dumps(theDict))
def read_dict_float(strang):
  # reading the data from the file
  with open(strang) as f:
      data = f.read()
  js = json.loads(data)
  return js
def connected_component_subgraphs(G):
    G=copy.deepcopy(G)
    G=G.to_undirected()
    for c in nx.connected_components(G):
        yield G.subgraph(c)
def count_all_triangles(G,node):
  G=nx.DiGraph(G)
  G.remove_edges_from(list(nx.selfloop_edges(G)))  
  return ((1/2)*(nx.to_numpy_matrix(G)+np.transpose(nx.to_numpy_matrix(G)))**3)[node,node]

def average_all_triangles(G):
  G=nx.DiGraph(G)
  G.remove_edges_from(list(nx.selfloop_edges(G)))
  return np.mean([count_all_triangles(G,i) for i in G.nodes()])

def all_triangles_clustering(G,node):
  G=nx.DiGraph(G)
  G.remove_edges_from(list(nx.selfloop_edges(G)))
  return ((1/2)*(nx.to_numpy_matrix(G)+np.transpose(nx.to_numpy_matrix(G)))**3)[node,node]/(G.degree(node)*(G.degree(node)-1)-2*(nx.to_numpy_matrix(G)**2)[node,node])

def reciprocal_degree(G,node):
  G.remove_edges_from(list(nx.selfloop_edges(G)))
  return (nx.to_numpy_matrix(G)**2)[node,node]
def proportion_of_zero_out_degree_nodes(G):
  return len([i for i in G.nodes() if G.out_degree(i)==0])/len(G.nodes())
def max_betweenness_centrality(G):
  G=nx.DiGraph(G)
  G.remove_edges_from(list(nx.selfloop_edges(G)))
  return max(nx.betweenness_centrality(G).values())
  
  
def max_eigenvector_centrality(G):
  G=nx.DiGraph(G)
  G.remove_edges_from(list(nx.selfloop_edges(G)))
  return max(nx.eigenvector_centrality(G).values())

def fiedler_value(G):
  G=nx.DiGraph(G)
  G.remove_edges_from(list(nx.selfloop_edges(G)))
  return nx.laplacian_spectrum(G)[1]
def average_all_triangles_clustering(G):
  G=nx.DiGraph(G)
  G.remove_edges_from(list(nx.selfloop_edges(G)))
  return np.mean([all_triangles_clustering(G,i) for i in G.nodes()])
def count_FFL_triangles(G,node):
  G=nx.DiGraph(G)
  G.remove_edges_from(list(nx.selfloop_edges(G)))
  return (nx.to_numpy_matrix(G)*np.transpose(nx.to_numpy_matrix(G)*nx.to_numpy_matrix(G)))[node,node]

def average_FFL_triangles_igraph(G):
  '''return the number of ffl triangles for igraph instead of networkx'''
  
  A=np.array(G.get_adjacency().data)
  np.fill_diagonal(A,0)
  matmult=np.matmul(np.matmul(A,np.transpose(A)),A)

  return np.mean([matmult[i,i] for i in range(G.vcount())])

def number_of_children_that_share_a_child(G,node):
  return len([i for i in G.successors(node) if len(set(G.successors(i)).intersection(set(G.successors(node))))>0])
nx.transitivity
def FFL_triangles_clustering_igraph(G):
  '''return the number of ffl triangles for igraph instead of networkx'''

  A=np.array(G.get_adjacency().data)
  np.fill_diagonal(A,0)
  matmult=np.matmul(np.matmul(A,np.transpose(A)),A)
  meanVec=[]
  for i in range(G.vcount()):
    degMult=G.degree(i,mode='out',loops=False)*G.degree(i,mode='in',loops=False)-reciprocal_degree_igraph(G,i)
    if degMult!=0:
      meanVec.append(matmult[i,i]/degMult)
  return np.mean(meanVec)
def FFL_triangles_of_a_node_igraph(G,i):
  '''return the number of ffl triangles for igraph instead of networkx'''
  count=0
  outNeigh=set(G.neighbors(i,mode='out')).difference(set([i]))
  for j in G.neighbors(i,mode='in'):
    count=count+len(outNeigh & set(G.neighbors(j,mode='out')))
      
  return count
   
def count_cycle_triangles(G,node):
  G=nx.DiGraph(G)
  G.remove_edges_from(list(nx.selfloop_edges(G)))
  return ((nx.to_numpy_matrix(G))**3)[node,node]

def average_cycle_triangles(G):
  return np.mean([count_cycle_triangles(G,i) for i in G.nodes()])

def cycle_triangles_clustering(G,node):
  G=nx.DiGraph(G)
  G.remove_edges_from(list(nx.selfloop_edges(G)))
  return ((nx.to_numpy_matrix(G))**3)[node,node]/(G.in_degree(node)*G.out_degree(node)-(nx.to_numpy_matrix(G)**2)[node,node])
def remove_self_loops(G):
    G_copy=G.copy()
    
    for e in [(i.source,i.target) for i in G_copy.es]:
        if e[0]==e[1]:
            
            G_copy.delete_edges(e)
    return G_copy
def average_cycle_triangles_clustering(G):
  G=nx.DiGraph(G)
  G.remove_edges_from(list(nx.selfloop_edges(G)))
  return np.mean([cycle_triangles_clustering(G,i) for i in G.nodes()])

def gnp_random_graph(n, p, seed=None, directed=False):
    """Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph
    or a binomial graph.

    The $G_{n,p}$ model chooses each of the possible edges with probability $p$.

    Parameters
    ----------
    n : int
        The number of nodes.
    p : float
        Probability for edge creation.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    directed : bool, optional (default=False)
        If True, this function returns a directed graph.

    See Also
    --------
    fast_gnp_random_graph

    Notes
    -----
    This algorithm [2]_ runs in $O(n^2)$ time.  For sparse graphs (that is, for
    small values of $p$), :func:`fast_gnp_random_graph` is a faster algorithm.

    :func:`binomial_graph` and :func:`erdos_renyi_graph` are
    aliases for :func:`gnp_random_graph`.

    >>> nx.binomial_graph is nx.gnp_random_graph
    True
    >>> nx.erdos_renyi_graph is nx.gnp_random_graph
    True

    References
    ----------
    .. [1] P. Erdős and A. Rényi, On Random Graphs, Publ. Math. 6, 290 (1959).
    .. [2] E. N. Gilbert, Random Graphs, Ann. Math. Stat., 30, 1141 (1959).
    """
    if directed:
        edges = list(itertools.permutations(range(n), 2))
        #for i in range(n):
        #  edges.append((i,i))
          
        G = nx.DiGraph()
    else:
        edges = itertools.combinations(range(n), 2)
        G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    

    for e in edges:
        rando=np.random.rand(1)
        if rando < p:
            G.add_edge(*e)
    return G

def kperm(k, n, exclude=[], init=None):
	if init is None: 
		init = k
	if k == init-2:
		for i in range(0, n):
			if i not in exclude:
				yield (i,)
	for firstnum in range(0,n):
		if firstnum not in exclude:
			for x in kperm(k-1, n, exclude=exclude+[firstnum], init=init):
				yield tuple((firstnum,) + x) 
def is_automorphic(G1,G2):
  if len(G1.nodes())!=len(G2.nodes()):
    return False
  for i in set(G1.edges()):
    if i not in set(G2.edges()):
      return False
  for i in set(G2.edges()):
    if i not in set(G1.edges()):
      return False
  return True
def is_automorphic_fast(G1,G2):
  #if len(G1.nodes())!=len(G2.nodes()):
  #  return False
  #if len(set(G1.nodes)^set(G2.nodes))!=0:
  #  return False
  if len(set(G1.edges)^set(G2.edges))!=0:
    return False
  return True
def ped_is_automorphic(G1,G2,i1,i2):
  if len(set(G1.predecessors(i1))^set(G2.predecessors(i2)))!=0:
    return False
  if len(set(G1.successors(i1))^set(G2.successors(i2)))!=0:
    return False
  return True
def random_induced_subgraph(G,n):
  randomlist = random.sample(range(0, len(G.nodes)), n)
  nodeList=[list(G.nodes)[i] for i in randomlist]
  G_ind = nx.induced_subgraph(G,nodeList)
  return G_ind
def random_connected_induced_subgraph_naive(G,n):
  G_ind = random_induced_subgraph(G,n)
  while not nx.is_connected(nx.Graph(G_ind)):
    G_ind = random_induced_subgraph(G,n)
  return G_ind

def non_uniform_random_connected_induced_subgraph(G,n):
  rando= int(len(G.nodes) * random.random())
  start_node = list(G.nodes)[rando]
  G_ind = nx.Graph()
  nodeList=[]
  G_ind.add_node(start_node)
  nodeList.append(start_node)
  for j in range(0,n-1):
    neighbors=[]
    for k in nodeList:

      neighbors = neighbors+ [i for i in list(G.predecessors(k))+list(G.successors(k)) if i not in nodeList]
    rando= int(len(neighbors) * random.random())
    if rando==len(neighbors):
      return non_uniform_random_connected_induced_subgraph(G,n)
    nodeList.append(neighbors[rando])
  G_ind=nx.induced_subgraph(G,nodeList)
  
  return G_ind

def non_uniform_random_connected_induced_subgraph_igraph(G,n):
  rando= int(G.vcount() * random.random())
  start_node = G.vs[rando].index
  G_ind = ig.Graph()
  nodeList=[]
  #G_ind.add_vertex(start_node)
  nodeList.append(start_node)
  for j in range(0,n-1):
    neighbors=[]
    for k in nodeList:

      neighbors = neighbors+ [i for i in G.neighbors(k) if i not in nodeList]
    rando= int(len(neighbors) * random.random())
    if rando==len(neighbors):
      return non_uniform_random_connected_induced_subgraph_igraph(G,n)
    nodeList.append(neighbors[rando])
  
  
  return nodeList

def missing_degree_of_subgraph(G,G_ind):
  
  missing_degree=0
  for i in G_ind.nodes():
    missing_degree=missing_degree+G.degree(i)-G_ind.degree(i)
  return missing_degree

def missing_degree_of_subgraph_igraph(G,subgraph):
  missing_degree=0
  for i in subgraph.vs:
    
    missing_degree=missing_degree+G.degree(G.vs.find(name=i['name']))-subgraph.degree(i)

  return missing_degree

def met_hastings_connected_induced_subgraph_igraph(G,n,conv):
  initNodeList=non_uniform_random_connected_induced_subgraph_igraph(G,n)
  #print(initNodeList)
  initGraph=G.induced_subgraph(initNodeList)
  d_i=missing_degree_of_subgraph_igraph(G,initGraph)
  iterr=0
  nodeList=initNodeList
  while iterr<conv:
    #nodeList=list(initGraph.vs)
    #print(nodeList)
    rando1= int(n * random.random()-1)
    #print('to del',nodeList[rando1])
    toDel=nodeList[rando1]
    nodeList.remove(toDel)
    #print(nodeList)
    neighbors=[]
    for k in nodeList:

        neighbors = neighbors+ [i for i in G.neighbors(k) if i not in nodeList]
    rando2= int(len(neighbors) * random.random())
    if rando2==len(neighbors):
      print(rando2)
    #print("to add",neighbors[rando2])
    toAdd=neighbors[rando2]
    nodeList.append(toAdd)
    
    newGraph=G.induced_subgraph(nodeList)
    if newGraph.is_connected(mode='weak'):
      #print(nodeList)
      d_j=missing_degree_of_subgraph_igraph(G,newGraph)
      #print(d_j)
      rando3=random.random()
      if rando3<d_i/d_j:
          initGraph=newGraph
          d_i=d_j
        
      iterr=iterr+1
    else:
     
      nodeList.remove(toAdd)
      nodeList.append(toDel)
  return initGraph

def met_hastings_alignment(G1,G2,align,conv):
  #initNodeList=non_uniform_random_connected_induced_subgraph_igraph(G,n)
  #print(initNodeList)
  #initGraph=G.induced_subgraph(initNodeList)
  #d_i=missing_degree_of_subgraph_igraph(G,initGraph)
  tempAlign=align.copy()
  n=len(tempAlign[0])
  iterr=0
  #nodeList=initNodeList
  while iterr<conv:
    #nodeList=list(initGraph.vs)
    #print(nodeList)
    rando1= int(n * random.random()-1)
    #print('to del',nodeList[rando1])
    
    tempAlign[0].remove(tempAlign[0][rando1])
    tempAlign[1].remove(tempAlign[1][rando1])
    #print(nodeList)
    mappingList=[list(set(range(G1.vcount())).difference(set(tempAlign[0]))),list(set(range(G1.vcount())).difference(set(tempAlign[0])))]
    #for k in range(0,len(mappingList[0])):
    #    for j in range(0,len(mappingList[1])):
    #        

    #    neighbors = neighbors+ [i for i in G.neighbors(k) if i not in nodeList]
    rando2= int(len(mappingList[0]) * random.random())
    rando3= int(len(mappingList[1]) * random.random())
    tempAlign[0].append(mappingList[0][rando2])
    tempAlign[1].append(mappingList[1][rando3])

    
   
    if newGraph.is_connected(mode='weak'):
      #print(nodeList)
      d_j=missing_degree_of_subgraph_igraph(G,newGraph)
      #print(d_j)
      rando3=random.random()
      if rando3<d_i/d_j:
          initGraph=newGraph
          d_i=d_j
        
      iterr=iterr+1
    else:
     
      nodeList.remove(toAdd)
      nodeList.append(toDel)
  return initGraph

def met_hastings_connected_induced_subgraph(G,n,conv,mix):
  initGraph=non_uniform_random_connected_induced_subgraph(G,n)
  d_i=missing_degree_of_subgraph(G,initGraph)
  iterr=0
  while iterr<conv:
    nodeList=list(initGraph.nodes)
    
    rando1= int(n * random.random())
    
    nodeList.remove(list(initGraph.nodes)[rando1])

    neighbors=[]
    for k in nodeList:

        neighbors = neighbors+ [i for i in list(G.predecessors(k))+list(G.successors(k)) if i not in nodeList]
    rando2= int(len(neighbors) * random.random())
    if rando2==len(neighbors):
      print(rando2)
    nodeList.append(neighbors[rando2])
    if nx.is_connected(nx.Graph(nx.induced_subgraph(G,nodeList))):
      d_j=missing_degree_of_subgraph(G,nx.induced_subgraph(G,nodeList))
      rando3=random.random()
      if rando3<d_i/d_j:
        initGraph=nx.induced_subgraph(G,nodeList)
        d_i=d_j
      
      iterr=iterr+1
  return initGraph
def degree_distribution_distance(G1,G2):
  d1=nx.degree_histogram(G1)
  d2=nx.degree_histogram(G2)
  return sum([abs(d1[i]-d2[i]) for i in range(0,len(d1))])
  #return np.sum(np.abs(d1-d2))

def average_degree(G):
  return sum([d for (n, d) in nx.degree(G)]) / len(G.nodes())
def cytoscape_graph(data, attrs=None, name="name", ident="id",value="id"):
    """
    Create a NetworkX graph from a dictionary in cytoscape JSON format.

    Parameters
    ----------
    data : dict
        A dictionary of data conforming to cytoscape JSON format.
    attrs : dict or None (default=None)
        A dictionary containing the keys 'name' and 'ident' which are mapped to
        the 'name' and 'id' node elements in cyjs format. All other keys are
        ignored. Default is `None` which results in the default mapping
        ``dict(name="name", ident="id")``.

        .. deprecated:: 2.6

           The `attrs` keyword argument will be replaced with `name` and
           `ident` in networkx 3.0

    name : string
        A string which is mapped to the 'name' node element in cyjs format.
        Must not have the same value as `ident`.
    ident : string
        A string which is mapped to the 'id' node element in cyjs format.
        Must not have the same value as `name`.

    Returns
    -------
    graph : a NetworkX graph instance
        The `graph` can be an instance of `Graph`, `DiGraph`, `MultiGraph`, or
        `MultiDiGraph` depending on the input data.

    Raises
    ------
    NetworkXError
        If the `name` and `ident` attributes are identical.

    See Also
    --------
    cytoscape_data: convert a NetworkX graph to a dict in cyjs format

    References
    ----------
    .. [1] Cytoscape user's manual:
       http://manual.cytoscape.org/en/stable/index.html

    Examples
    --------
    >>> data_dict = {
    ...     'data': [],
    ...     'directed': False,
    ...     'multigraph': False,
    ...     'elements': {'nodes': [{'data': {'id': '0', 'value': 0, 'name': '0'}},
    ...       {'data': {'id': '1', 'value': 1, 'name': '1'}}],
    ...      'edges': [{'data': {'source': 0, 'target': 1}}]}
    ... }
    >>> G = nx.cytoscape_graph(data_dict)
    >>> G.name
    ''
    >>> G.nodes()
    NodeView((0, 1))
    >>> G.nodes(data=True)[0]
    {'id': '0', 'value': 0, 'name': '0'}
    >>> G.edges(data=True)
    EdgeDataView([(0, 1, {'source': 0, 'target': 1})])
    """
    # ------ TODO: Remove between the lines in 3.0 ----- #
    if attrs is not None:
        import warnings

        msg = (
            "\nThe `attrs` keyword argument of cytoscape_data is deprecated\n"
            "and will be removed in networkx 3.0.\n"
            "It is replaced with explicit `name` and `ident` keyword\n"
            "arguments.\n"
            "To make this warning go away and ensure usage is forward\n"
            "compatible, replace `attrs` with `name` and `ident`,\n"
            "for example:\n\n"
            "   >>> cytoscape_data(G, attrs={'name': 'foo', 'ident': 'bar'})\n\n"
            "should instead be written as\n\n"
            "   >>> cytoscape_data(G, name='foo', ident='bar')\n\n"
            "The default values of 'name' and 'id' will not change."
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

        name = attrs["name"]
        ident = attrs["ident"]
    # -------------------------------------------------- #

    if name == ident:
        raise nx.NetworkXError("name and ident must be different.")

    multigraph = data.get("multigraph")
    directed = data.get("directed")
    if multigraph:
        graph = nx.MultiGraph()
    else:
        graph = nx.Graph()
    if directed:
        graph = graph.to_directed()
    graph.graph = dict(data.get("data"))
    for d in data["elements"]["nodes"]:
        node_data = d["data"].copy()
        node = d["data"][value]

        if d["data"].get(name):
            node_data[name] = d["data"].get(name)
        if d["data"].get(ident):
            node_data[ident] = d["data"].get(ident)

        graph.add_node(node)
        graph.nodes[node].update(node_data)

    for d in data["elements"]["edges"]:
        edge_data = d["data"].copy()
        sour = d["data"]["source"]
        targ = d["data"]["target"]
        if multigraph:
            key = d["data"].get("key", 0)
            graph.add_edge(sour, targ, key=key)
            graph.edges[sour, targ, key].update(edge_data)
        else:
            graph.add_edge(sour, targ)
            graph.edges[sour, targ].update(edge_data)
    return graph

  
def directed_double_edge_swap(G, nswap=1, max_tries=100, seed=None,self_loops=False):
    """Swap two edges in the graph while keeping the node degrees fixed.

    A double-edge swap removes two randomly chosen edges u-v and x-y
    and creates the new edges u-x and v-y::

     u--v            u  v
            becomes  |  |
     x--y            x  y

    If either the edge u-x or v-y already exist no swap is performed
    and another attempt is made to find a suitable edge pair.

    Parameters
    ----------
    G : graph
       An undirected graph

    nswap : integer (optional, default=1)
       Number of double-edge swaps to perform

    max_tries : integer (optional)
       Maximum number of attempts to swap edges

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : graph
       The graph after double edge swaps.

    Notes
    -----
    Does not enforce any connectivity constraints.

    The graph G is modified in place.
    """
    
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G) < 4:
        return("graph too small")
    # Instead of choosing uniformly at random from a generated edge list,
    # this algorithm chooses nonuniformly from the set of nodes with
    # probability weighted by degree.
    n = 0
    swapcount = 0
    keys, degrees = zip(*G.degree())  # keys, degree
    cdf = nx.utils.cumulative_distribution(degrees)  # cdf of degree
    discrete_sequence = nx.utils.discrete_sequence
    while swapcount < nswap:
        #        if random.random() < 0.5: continue # trick to avoid periodicities?
        # pick two random edges without creating edge list
        # choose source node indices from discrete distribution
        (ui, xi) = discrete_sequence(2, cdistribution=cdf, seed=seed)
        if ui == xi:
            continue  # same source, skip
        u = keys[ui]  # convert index to label
        x = keys[xi]
        # choose target uniformly from neighbors
        uNeigh=list(G.predecessors(u)) + list(G.successors(u))
        
        xNeigh=list(G.predecessors(x)) + list(G.successors(x))
        

        v = random.choice(uNeigh)
        y = random.choice(xNeigh)
        vNeigh=list(G.predecessors(v)) + list(G.successors(v))
        
        yNeigh=list(G.predecessors(y)) + list(G.successors(y))
        
        if (not self_loops) and (v == y or u == x or y==u or x==v):
            continue  # same target, skip

        if (u,v) in G.edges() and (v,u) not in G.edges():
          if (x,y) in G.edges() and (y,x) not in G.edges():
            if (y not in G.successors(u)) and (v not in G.successors(x)):  # don't create parallel edges
                G.add_edge(u, y)
                G.add_edge(x, v)
                G.remove_edge(u, v)
                G.remove_edge(x, y)
                swapcount += 1
          elif (x,y) not in G.edges() and (y,x) in G.edges():
            if (x not in G.successors(u)) and (v not in G.successors(y)):  # don't create parallel edges
                G.add_edge(u, x)
                G.add_edge(y, v)
                G.remove_edge(u, v)
                G.remove_edge(y, x)
                swapcount += 1
          elif (x,y) in G.edges() and (y,x) in G.edges():
            rand=random.random()
            if rand>0.5:
              if (y not in G.successors(u)) and (v not in G.successors(x)):  # don't create parallel edges
                G.add_edge(u, y)
                G.add_edge(x, v)
                G.remove_edge(u, v)
                G.remove_edge(x, y)
                swapcount += 1
            else:
              if (x not in G.successors(u)) and (v not in G.successors(y)):  # don't create parallel edges
                G.add_edge(u, x)
                G.add_edge(y, v)
                G.remove_edge(u, v)
                G.remove_edge(y, x)
                swapcount += 1
        elif (u,v) not in G.edges() and (v,u) in G.edges():
          if (x,y) in G.edges() and (y,x) not in G.edges():
            if (u not in G.successors(x)) and (y not in G.successors(v)):  # don't create parallel edges
                G.add_edge(x, u)
                G.add_edge(v, y)
                G.remove_edge(v, u)
                G.remove_edge(x, y)
                swapcount += 1
          elif (x,y) not in G.edges() and (y,x) in G.edges():
            if (u not in G.successors(y)) and (x not in G.successors(v)):  # don't create parallel edges
                G.add_edge(y, u)
                G.add_edge(v, x)
                G.remove_edge(v, u)
                G.remove_edge(y, x)
                swapcount += 1
          elif (x,y) in G.edges() and (y,x) in G.edges():
            rand=random.random()
            if rand>0.5:
              if (u not in G.successors(x)) and (y not in G.successors(v)):  # don't create parallel edges
                G.add_edge(x, u)
                G.add_edge(v, y)
                G.remove_edge(v, u)
                G.remove_edge(x, y)
                swapcount += 1
            else:
              if (u not in G.successors(y)) and (x not in G.successors(v)):  # don't create parallel edges
                G.add_edge(y, u)
                G.add_edge(v, x)
                G.remove_edge(v, u)
                G.remove_edge(y, x)
                swapcount += 1
        elif (u,v) in G.edges() and (v,u) in G.edges():
          rand = random.random()
          if rand>0.5:
            if (x,y) in G.edges() and (y,x) not in G.edges():
              if (y not in G.successors(u)) and (v not in G.successors(x)):  # don't create parallel edges
                  G.add_edge(u, y)
                  G.add_edge(x, v)
                  G.remove_edge(u, v)
                  G.remove_edge(x, y)
                  swapcount += 1
            elif (x,y) not in G.edges() and (y,x) in G.edges():
              if (x not in G.successors(u)) and (v not in G.successors(y)):  # don't create parallel edges
                  G.add_edge(u, x)
                  G.add_edge(y, v)
                  G.remove_edge(u, v)
                  G.remove_edge(y, x)
                  swapcount += 1
            elif (x,y) in G.edges() and (y,x) in G.edges():
              rand=random.random()
              if rand>0.5:
                if (y not in G.successors(u)) and (v not in G.successors(x)):  # don't create parallel edges
                  G.add_edge(u, y)
                  G.add_edge(x, v)
                  G.remove_edge(u, v)
                  G.remove_edge(x, y)
                  swapcount += 1
              else:
                if (x not in G.successors(u)) and (v not in G.successors(y)):  # don't create parallel edges
                  G.add_edge(u, x)
                  G.add_edge(y, v)
                  G.remove_edge(u, v)
                  G.remove_edge(y, x)
                  swapcount += 1
          else:
            if (x,y) in G.edges() and (y,x) not in G.edges():
              if (u not in G.successors(x)) and (y not in G.successors(v)):  # don't create parallel edges
                  G.add_edge(x, u)
                  G.add_edge(v, y)
                  G.remove_edge(v, u)
                  G.remove_edge(x, y)
                  swapcount += 1
            elif (x,y) not in G.edges() and (y,x) in G.edges():
              if (u not in G.successors(y)) and (x not in G.successors(v)):  # don't create parallel edges
                  G.add_edge(y, u)
                  G.add_edge(v, x)
                  G.remove_edge(v, u)
                  G.remove_edge(y, x)
                  swapcount += 1
            elif (x,y) in G.edges() and (y,x) in G.edges():
              rand=random.random()
              if rand>0.5:
                if (u not in G.successors(x)) and (y not in G.successors(v)):  # don't create parallel edges
                  G.add_edge(x, u)
                  G.add_edge(v, y)
                  G.remove_edge(v, u)
                  G.remove_edge(x, y)
                  swapcount += 1
              else:
                if (u not in G.successors(y)) and (x not in G.successors(v)):  # don't create parallel edges
                  G.add_edge(y, u)
                  G.add_edge(v, x)
                  G.remove_edge(v, u)
                  G.remove_edge(y, x)
                  swapcount += 1
        if n >= max_tries:
            e = (
                f"Maximum number of swap attempts ({n}) exceeded "
                f"before desired swaps achieved ({nswap})."
            )
            return("graph too small")
        n += 1
    return G

def graph_intersection_union(G1,G2):
  G1_temp=copy.deepcopy(G1)
  G2_temp=copy.deepcopy(G2)
  G_intersect=nx.intersection(G1_temp,G2_temp)
  G1_induced=nx.induced_subgraph(G1_temp,G_intersect.nodes())
  G2_induced=nx.induced_subgraph(G2_temp,G_intersect.nodes())
  for i in G1_induced.edges():
    if i not in G_intersect.edges():
      G_intersect.add_edge(i[0],i[1])
  for i in G2_induced.edges():
    if i not in G_intersect.edges():
      G_intersect.add_edge(i[0],i[1])
  return G_intersect

def proportion_of_nodes_with_selfloops(G):
  
  return len([(i,j) for (i,j) in G.edges() if i==j])/len(G.nodes())
def average_degree(G):
  return sum([G.degree(i) for i in G.nodes()])/len(G.nodes())
def average_outdegree(G):
  return sum([G.out_degree(i) for i in G.nodes()])/len(G.nodes())
def average_indegree(G):
  return sum([G.in_degree(i) for i in G.nodes()])/len(G.nodes())

def null_distribution_s3_score(G_random,G_base,resolution=100):
    scoreVec=[]
    edges=len(G_random.edges)
    
    G=copy.deepcopy(G_random)
    i=0
    while len(scoreVec)<resolution:
        
        i=i+1
        
        G=directed_double_edge_swap(G,1)
        if i%edges==0:
            score=s3_score(G,G_base)
            #print(i,"/",resolution*edges,score)
            scoreVec.append(score)
        
    return scoreVec
def null_distribution_ec_score(G_random,G_base,resolution=100000):
    scoreVec=[]
    edges=len(G_random.edges)
    
    G=copy.deepcopy(G_random)
    i=0
    while len(scoreVec)<resolution:
        
        i=i+1
        
        G=directed_double_edge_swap(G,1)
        if isinstance(G,str):
            print("graph too small")
            return scoreVec
        if i%(2*edges)==0:
            score=normalised_ec_score(G,G_base)
            #print(i,"/",resolution*edges,score)
            scoreVec.append(score)
        
    return scoreVec
def random_number_of_conserved_edges_mean(n,deg_seq_out1,deg_seq_out2,deg_seq_in1,deg_seq_in2,self_loops=True):
  summ=0
  m1=sum(deg_seq_in1)
  m2=sum(deg_seq_in2)
  if self_loops:
    for i in range(n):
      for j in range(n):
        summ=summ+deg_seq_out1[i]*deg_seq_out2[i]*deg_seq_in1[j]*deg_seq_in2[j]/(m1*m2)
  else:
    for i in range(n):
      for j in range(n-1):
        summ=summ+deg_seq_out1[i]*deg_seq_out2[i]*deg_seq_in1[j]*deg_seq_in2[j]/((m1-deg_seq_in1[i])*(m2-deg_seq_in2[i]))
  return summ
def random_number_of_conserved_edges_std(n,deg_seq_out1,deg_seq_out2,deg_seq_in1,deg_seq_in2,self_loops=True):
  summ=0
  m1=sum(deg_seq_in1)
  m2=sum(deg_seq_in2)
  if self_loops:
    for i in range(n):
      for j in range(n):
        summ=summ+(deg_seq_out1[i]*deg_seq_out2[i]*deg_seq_in1[j]*deg_seq_in2[j]/(m1*m2))*(1-deg_seq_out1[i]*deg_seq_out2[i]*deg_seq_in1[j]*deg_seq_in2[j]/(m1*m2))
  else:
    for i in range(n):
      for j in range(n-1):
        summ=summ+(deg_seq_out1[i]*deg_seq_out2[i]*deg_seq_in1[j]*deg_seq_in2[j]/((m1-deg_seq_in1[i])*(m2-deg_seq_in2[i])))*(1-deg_seq_out1[i]*deg_seq_out2[i]*deg_seq_in1[j]*deg_seq_in2[j]/((m1-deg_seq_in1[i])*(m2-deg_seq_in2[i])))
  if summ>=0:    
    return np.sqrt(summ)
  else:
    print("error in std")
    return 1

def getMCS(G_source, G_new):
  #USER: BONSON STACK OVERFLOW MAY 9 2017
    matching_graph=nx.Graph()

    for n1,n2,attr in G_new.edges(data=True):
        if G_source.has_edge(n1,n2) :
            matching_graph.add_edge(n1,n2,weight=1)

    graphs = list(connected_component_subgraphs(matching_graph))

    mcs_length = 0
    mcs_graph = nx.Graph()
    for i, graph in enumerate(graphs):

        if len(graph.nodes()) > mcs_length:
            mcs_length = len(graph.nodes())
            mcs_graph = graph

    return mcs_graph
def geo_mean(iterable):

    a = np.array(iterable)
    return a.prod()**(1.0/len(a))
def takeThird(elem):
  return elem[2]
def degree_sequence_plot(G):
  out_deg_dist=dict()
  in_deg_dist=dict()
  for i in G.nodes:
      in_deg_dist[i]=0
      out_deg_dist[i]=0
      for j in G.predecessors(i):
          in_deg_dist[i]=in_deg_dist[i]+1
      for j in G.successors(i):
          out_deg_dist[i]=out_deg_dist[i]+1

  inList=list(in_deg_dist.values())
  outList=list(out_deg_dist.values())
  inList=sorted(inList)
  outList=sorted(outList)
  
  bins = np.arange(0, max(outList), 1) # fixed bin size

  plt.hist(outList, bins=bins, alpha=0.5)

  plt.title('')
  plt.xlabel('Degree')
  plt.ylabel('Count')

  plt.show()
  plt.figure()

  bins = np.arange(0, max(inList), 1) # fixed bin size

  plt.hist(inList, bins=bins, alpha=0.5)

  plt.title('')
  plt.xlabel('Degree')
  plt.ylabel('Count')

  plt.show()
def loglog_degree_sequence_plot(G):
  out_deg_dist=dict()
  in_deg_dist=dict()
  for i in G.nodes:
      in_deg_dist[i]=0
      out_deg_dist[i]=0
      for j in G.predecessors(i):
          in_deg_dist[i]=in_deg_dist[i]+1
      for j in G.successors(i):
          out_deg_dist[i]=out_deg_dist[i]+1

  inList=list(in_deg_dist.values())
  outList=list(out_deg_dist.values())
  inList=sorted(inList)
  outList=sorted(outList)
  bins = np.arange(0, max(outList), 1) # fixed bin size

  hist, bins = np.histogram(outList, bins=bins)
  logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
  plt.hist(outList, bins=bins, alpha=0.5,log=True)
  plt.xscale('log')
  plt.title('')
  plt.xlabel('Degree')
  plt.ylabel('Count')

  plt.show()
  bins = np.arange(0, max(inList), 1) # fixed bin size
  plt.figure()
  hist, bins = np.histogram(inList, bins=bins)
  logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
  plt.hist(inList, bins=bins, alpha=0.5,log=True)
  plt.xscale('log')
  plt.title('')
  plt.xlabel('Degree')
  plt.ylabel('Count')

  plt.show()

def pair_loglog_degree_sequence_plot(G1,G2):
  for G in [G1,G2]:
    out_deg_dist=dict()
    in_deg_dist=dict()
    for i in G.nodes:
        
        in_deg_dist[i]=0.01
        out_deg_dist[i]=0.01
        for j in G.predecessors(i):
            in_deg_dist[i]=in_deg_dist[i]+1
        for j in G.successors(i):
            out_deg_dist[i]=out_deg_dist[i]+1

    inList=list(in_deg_dist.values())
    outList=list(out_deg_dist.values())
    inList=sorted(inList)
    outList=sorted(outList)
    if G==G2:
      print(inList,outList)
    bins = np.arange(0.01, max(outList), 1) # fixed bin size

    hist, bins = np.histogram(outList, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(outList, bins=bins, alpha=0.5,log=True)
    plt.xscale('log')
    plt.title('')
    plt.xlabel('Degree')
    plt.ylabel('Count')

    
  
  plt.figure()
  for G in [G1,G2]:
    out_deg_dist=dict()
    in_deg_dist=dict()
    for i in G.nodes:
        in_deg_dist[i]=0.01
        out_deg_dist[i]=0.01
        for j in G.predecessors(i):
            in_deg_dist[i]=in_deg_dist[i]+1
        for j in G.successors(i):
            out_deg_dist[i]=out_deg_dist[i]+1

    inList=list(in_deg_dist.values())
    outList=list(out_deg_dist.values())
    inList=sorted(inList)
    outList=sorted(outList)
    bins = np.arange(0.01, max(inList), 1) # fixed bin size
    hist, bins = np.histogram(inList, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(inList, bins=bins, alpha=0.5,log=True)
    plt.xscale('log')
    plt.title('')
    plt.xlabel('Degree')
    plt.ylabel('Count')


def local_reaching_centrality_igraph(G, v, paths=None, weight=None, normalized=True):
    '''returns local reaching centrality for igraph instead of networkx'''
    summ=0
    if paths is None:
        paths = single_source_shortest_path(G,v)
        
        for i in paths:
               summ=summ+1
        
        return (summ - 1) / (G.vcount() - 1)
    else:
      return (len(paths) - 1) / (G.vcount() - 1)
def local_reaching_centrality_undirected_igraph(G, v, paths=None, weight=None, normalized=True):
   summ=0
   if paths is None:
        paths = single_source_shortest_path(G,v)
        for i in paths:
            summ=summ+1/len(paths[i])
        return (summ - 1) / (G.vcount() - 1)
   else:
      for i in paths:
            summ=summ+1/len(paths[i])
      return (summ - 1) / (G.vcount() - 1)
def global_reaching_centrality_igraph(G,weight=None,shortest_paths=None,undirected=False):
  '''returns global reaching centrality for igraph instead of networkx'''
  
  if shortest_paths is None:
     
    shortest_paths = shortest_path(G)
  if undirected:
    centrality = local_reaching_centrality_undirected_igraph
  else:
    centrality = local_reaching_centrality_igraph
   
  lrc = [centrality(G, node, paths=paths, weight=weight) for node, paths in shortest_paths.items()]
  
  max_lrc = max(lrc)
  return [sum(max_lrc - c for c in lrc) / (G.vcount() - 1),max_lrc]

def shortest_path(G, source=None, target=None, weight=None, method="unweighted"):
    '''this, and all subsequent functions are modified from networkx to work with igraph'''
    if method == "unweighted":
      paths = dict(all_pairs_shortest_path(G))
    else:
       raise ValueError(f"method not supported: {method}")
           
    return paths


def all_pairs_shortest_path(G, cutoff=None):
    
    for n in range(G.vcount()):
        yield (n, single_source_shortest_path(G, n, cutoff=cutoff))


def single_source_shortest_path(G, source, cutoff=None):
    
    
    def join(p1, p2):
        return p1 + p2

    if cutoff is None:
        cutoff = float("inf")
    nextlevel = {source: 1}  # list of nodes to check at next level
    paths = {source: [source]}  # paths dictionary  (paths to key from source)
    
    return dict(_single_shortest_path(G, nextlevel, paths, cutoff, join))


def _single_shortest_path(G, firstlevel, paths, cutoff, join):
   
    level = 0  # the current level
    nextlevel = firstlevel
    while nextlevel and cutoff > level:
        thislevel = nextlevel
        nextlevel = {}
        for v in thislevel:
            
              for w in G.neighbors(v,mode='out'):
                  if w not in paths:
                      paths[w] = join(paths[v], [w])
                      nextlevel[w] = 1
            
        level += 1
    return paths
def reciprocal_degree_igraph(G, node,selfloops=False):
  '''returns number of edges connected to node that are both incoming and outgoing for igraph instead of networkx'''
  if selfloops:
    return len(set(G.neighbors(node,mode='in')) & set(G.neighbors(node,mode='out')))

  else:
    inNeigh=set(G.neighbors(node,mode='in'))
    if node in inNeigh:
      return len( inNeigh & set(G.neighbors(node,mode='out')))-1
    else:
      return len( inNeigh & set(G.neighbors(node,mode='out')))
    
'''def reciprocal_degree_igraph(G,node):
  
  A=np.array(G.get_adjacency().data)
  np.fill_diagonal(A,0)
  matmult=np.matmul(A,A)

  return matmult[node,node]
'''
def average_out_degree_igraph(G,pre_out_degree=None):
  '''returns average out degree for igraph instead of networkx'''
  if pre_out_degree is None:
    pre_out_degree=G.outdegree()
  return sum(pre_out_degree)/G.vcount()

def average_in_degree_igraph(G,pre_in_degree=None):
  '''returns average in degree for igraph instead of networkx'''
  if pre_in_degree is None:
    pre_in_degree=G.indegree()
  return sum(pre_in_degree)/G.vcount()

def average_degree_igraph(G,pre_degree=None):
  '''returns average degree for igraph instead of networkx'''
  if pre_degree is None:
    pre_degree=G.degree()
  return sum(pre_degree)/G.vcount()
def degree_variance_igraph(G,pre_degree=None):
  '''returns degree variance for igraph instead of networkx'''
  if pre_degree is None:
    pre_degree=G.degree()
  return np.std(pre_degree,ddof=1)
def out_degree_variance_igraph(G,pre_out_degree=None):
  '''returns out degree variance for igraph instead of networkx'''
  if pre_out_degree is None:
    pre_out_degree=G.outdegree()
  return np.std(pre_out_degree,ddof=1)
def proportion_of_nodes_that_have_degree_one(G,pre_degree=None):
  '''returns proportion of nodes that have degree one for igraph instead of networkx'''
  if pre_degree is None:
    pre_degree=G.degree()
  return len([i for i in pre_degree if i==1])/G.vcount()
def in_degree_variance_igraph(G,pre_in_degree=None):
  '''returns in degree variance for igraph instead of networkx'''
  if pre_in_degree is None:
    pre_in_degree=G.indegree()
  return np.std(pre_in_degree,ddof=1)
def proportion_of_nodes_that_have_out_degree_zero(G,pre_out_degree=None):
  '''returns proportion of nodes that have out degree zero for igraph instead of networkx'''
  if pre_out_degree is None:
    pre_out_degree=G.outdegree()
  return len([i for i in pre_out_degree if i==0])/G.vcount()
def number_of_isolates(G,proportion=False):
  '''returns number of isolates for igraph instead of networkx'''
  if proportion:
    return len([i for i in list(G.degree()) if i==0])/G.vcount()
  else:
    return len([i for i in list(G.degree()) if i==0])
def proportion_of_nodes_that_have_self_loops(G):
  '''returns proportion of nodes that have self loops for igraph instead of networkx'''
  return sum([G.is_loop(i) for i in G.es()])/G.vcount()

def proportion_of_edges_that_have_reciprocal_edges(G):
  '''returns proportion of edges that have reciprocal edges for igraph instead of networkx'''
  return sum([G.is_mutual(i) for i in G.es])/G.ecount()

def maximum_local_reaching_centrality_igraph(G):
  '''returns maximum local reaching centrality for igraph instead of networkx'''
  return max([local_reaching_centrality_igraph(G,i) for i in range(G.vcount())])

def spectral_radius(G):
  '''returns spectral radius for igraph instead of networkx'''
  return max(abs(np.linalg.eigvals(np.array(G.get_adjacency().data))))

def in_degree_out_degree_difference_mean(G,pre_in_degree=None,pre_out_degree=None):
  '''returns average in degree out degree difference for igraph instead of networkx'''
  if pre_in_degree is None:
    pre_in_degree=G.indegree()
  if pre_out_degree is None:
    pre_out_degree=G.outdegree()
  vec=[]
  for i in range(G.vcount()):
     vec.append(np.abs(pre_in_degree[i]-pre_out_degree[i]))

  return np.mean(vec)

def in_degree_out_degree_difference_std(G,pre_in_degree=None,pre_out_degree=None):
  '''returns average in degree out degree difference for igraph instead of networkx'''
  if pre_in_degree is None:
    pre_in_degree=G.indegree()
  if pre_out_degree is None:
    pre_out_degree=G.outdegree()
  vec=[]
  for i in range(G.vcount()):
     vec.append(pre_in_degree[i]-pre_out_degree[i])

  return np.std(vec,ddof=1)
def harmonic_centrality_igraph(G):
  '''returns harmonic centrality for igraph instead of networkx'''
  return np.mean(G.harmonic_centrality(mode='out'))
def fit_power_law_exponent(G,deg_type='both'):

  if deg_type=='out':
    return ig.power_law_fit(G.outdegree()).alpha
  if deg_type=='in':
    return ig.power_law_fit(G.indegree()).alpha
  if deg_type=='both':
    return ig.power_law_fit(G.degree()).alpha

def all_pairs_node_connectivity(G):
    mapping = {}
    H = ig.Graph()

    for i, node in G.vs:
        mapping[node] = i
        H.add_vertex(f"{i}A")
        H.add_vertex(f"{i}B")
        H.add_edges([(f"{i}A", f"{i}B")])

    edges = []
    for i in G.es():
        edges.append((f"{mapping[i.source]}B", f"{mapping[i.target]}A"))
        
    H.add_edges(edges)

    # Store mapping as graph attribute
    H.graph["mapping"] = mapping

    iter_func = itertools.permutations
    

    all_pairs = {n: {} for n in nbunch}

    # Reuse auxiliary digraph and residual network
    
    R = build_residual_network(H, "capacity")
    kwargs = dict(flow_func=flow_func, auxiliary=H, residual=R)

    for u, v in iter_func(nbunch, 2):
        K = local_node_connectivity(G, u, v, **kwargs)
        all_pairs[u][v] = K

    return all_pairs

def local_node_connectivity(
    G, s, t, flow_func=None, auxiliary=None, residual=None, cutoff=None
):
  
    if flow_func is None:
        flow_func = default_flow_func

    if auxiliary is None:
        H = build_auxiliary_node_connectivity(G)
    else:
        H = auxiliary

    

    kwargs = dict(flow_func=flow_func, residual=residual)
    

    return nx.maximum_flow_value(H, f"{mapping[s]}B", f"{mapping[t]}A", **kwargs)

def build_residual_network(G, capacity):
    """Build a residual network and initialize a zero flow.

    The residual network :samp:`R` from an input graph :samp:`G` has the
    same nodes as :samp:`G`. :samp:`R` is a DiGraph that contains a pair
    of edges :samp:`(u, v)` and :samp:`(v, u)` iff :samp:`(u, v)` is not a
    self-loop, and at least one of :samp:`(u, v)` and :samp:`(v, u)` exists
    in :samp:`G`.

    For each edge :samp:`(u, v)` in :samp:`R`, :samp:`R[u][v]['capacity']`
    is equal to the capacity of :samp:`(u, v)` in :samp:`G` if it exists
    in :samp:`G` or zero otherwise. If the capacity is infinite,
    :samp:`R[u][v]['capacity']` will have a high arbitrary finite value
    that does not affect the solution of the problem. This value is stored in
    :samp:`R.graph['inf']`. For each edge :samp:`(u, v)` in :samp:`R`,
    :samp:`R[u][v]['flow']` represents the flow function of :samp:`(u, v)` and
    satisfies :samp:`R[u][v]['flow'] == -R[v][u]['flow']`.

    The flow value, defined as the total flow into :samp:`t`, the sink, is
    stored in :samp:`R.graph['flow_value']`. If :samp:`cutoff` is not
    specified, reachability to :samp:`t` using only edges :samp:`(u, v)` such
    that :samp:`R[u][v]['flow'] < R[u][v]['capacity']` induces a minimum
    :samp:`s`-:samp:`t` cut.

    """
    R = nx.DiGraph()
    R.add_nodes_from(G)

    inf = float("inf")
    # Extract edges with positive capacities. Self loops excluded.
    edge_list = [
        (u, v, attr)
        for u, v, attr in G.edges(data=True)
        if u != v and attr.get(capacity, inf) > 0
    ]
    # Simulate infinity with three times the sum of the finite edge capacities
    # or any positive value if the sum is zero. This allows the
    # infinite-capacity edges to be distinguished for unboundedness detection
    # and directly participate in residual capacity calculation. If the maximum
    # flow is finite, these edges cannot appear in the minimum cut and thus
    # guarantee correctness. Since the residual capacity of an
    # infinite-capacity edge is always at least 2/3 of inf, while that of an
    # finite-capacity edge is at most 1/3 of inf, if an operation moves more
    # than 1/3 of inf units of flow to t, there must be an infinite-capacity
    # s-t path in G.
    inf = (
        3
        * sum(
            attr[capacity]
            for u, v, attr in edge_list
            if capacity in attr and attr[capacity] != inf
        )
        or 1
    )
    if G.is_directed():
        for u, v, attr in edge_list:
            r = min(attr.get(capacity, inf), inf)
            if not R.has_edge(u, v):
                # Both (u, v) and (v, u) must be present in the residual
                # network.
                R.add_edge(u, v, capacity=r)
                R.add_edge(v, u, capacity=0)
            else:
                # The edge (u, v) was added when (v, u) was visited.
                R[u][v]["capacity"] = r
    else:
        for u, v, attr in edge_list:
            # Add a pair of edges with equal residual capacities.
            r = min(attr.get(capacity, inf), inf)
            R.add_edge(u, v, capacity=r)
            R.add_edge(v, u, capacity=r)

    # Record the value simulating infinity.
    R.graph["inf"] = inf

    return R



def nodal_eff(g):
    #g.es["weight"] = np.ones(g.ecount())
    #weights = g.es["weight"][:]
    short_dict=shortest_path(g)
    
    len_dict=[]
    len_dict=np.zeros((g.vcount(),g.vcount()))
    for i in short_dict:
        
        for j in short_dict[i]:
            len_dict[i][j]=len(short_dict[i][j])-1
    
    np.fill_diagonal(len_dict,1)
    sp = (1.0 / np.array(len_dict))
  
    np.fill_diagonal(sp,0)
    #N=sp.shape[0]
    #ne= (1.0/(N-1)) * np.apply_along_axis(sum,0,sp)

    return sp
def nodal_eff_std(G,pre_short=None):
  if pre_short is None:
    short_dict=shortest_path(G)
  else:
    short_dict=pre_short
  len_dict=[]
   
  for i in short_dict:
        
        for j in short_dict[i]:
            sam=len(short_dict[i][j])-1
            if sam!=0:
              len_dict.append(len(short_dict[i][j])-1)
    
   
  sp = [1.0 / i for i in len_dict]
  
  #node_eff=[]
  #for i in nodal_eff(G):
  #     for j in i:
  #         if j!=0:
  #             node_eff.append(j)
  return np.std(sp,ddof=1)

def edge_connectivity_std(G):
    edge_eff=[]
    for i in G.vs:
        for j in G.vs:
            if i!=j:
                edge_eff.append(G.edge_connectivity(i,j))
              
    return np.std(edge_eff,ddof=1)

def fit_q_degree_recurrence(G,G_seed,eps,r,iso_number=0):
  trueDeg=average_degree_igraph(G)
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

def degree_recurrence(G,G_seed,p,r,iso_number=0):
  #iso=len(list(nx.isolates(G_seed)))
  #SL=len(list(nx.selfloop_edges(G_seed)))
  deg=average_degree_igraph(G_seed)
  
  for i in range(len(list(G_seed.vs)),len(list(G.vs))+iso_number):
    summ=0
    
    deg=deg*(1+(1-2*p)/(i+1))
    #deg=deg*(1+(1-2*p)/(i+1)-(2*r)/(i*(i+1)))+(4*(1-p)*(SL/i))/(i+1)+(2*r)/(i+1)
    #SL=SL+(SL/i)*(1-p)
  
  #return ((i+1)*deg)/(i+1-iso_number)
  print("deg",deg,p)
  return deg
