# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
import sys 
import os
import itertools

import pandas as pd
import os
from tqdm import tqdm_notebook, tnrange
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import cvxpy as cp
from scipy.sparse import csr_matrix, vstack, hstack
from copy import deepcopy
module_path = os.path.abspath(os.path.join('..'))


def getReducedGraph(sample_nodes, graph_nodes, 
					interactome):

	"""
	Reduce graph with only intersection nodes from sample and
	interactome.

	Parameters:
	-----------
	sample_nodes : array-like, 
				   list of nodes/features found in dataset

	graph_nodes : array-like,
				  list of nodes/features found in graph/interactome

	interactome : Pandas dataframe
				  3 column dataframe [node1, node2, confidence] representing graph

	Returns:
	--------
	* : Pandas dataframe,
		Reduced subgraph that only contains nodes found in dataset

	"""

	#find intersection between sample nodes and graph nodes
	sample_nodes = set(sample_nodes)
	graph_nodes = set(graph_nodes)
	intersection_nodes = sample_nodes.intersection(graph_nodes)
	print('Number of Intersection Nodes : ', len(intersection_nodes))

	g = []
	for line in tqdm_notebook(range(len(interactome))):
		if (interactome.iloc[line]['node1'] in intersection_nodes
			and interactome.iloc[line]['node2'] in intersection_nodes):
			g.append(interactome.iloc[line])
	
	return pd.DataFrame(g)


def getNodeCharacterization(g, sample_nodes):
	"""
	Characterizes nodes based on if node is connected or orphan
	
	Parameters:
	-----------
	g : NetworkX graph, 
		reduced subgraph

	sample_nodes : array-like, 
				   list of nodes/features found in dataset


	Returns:
	--------
	connected_nodes : set,
					  nodes found in subgraph

	orphan_nodes : set,
				   nodes not found in subgraph
	"""

	connected_nodes = set(g.nodes())
	orphan_nodes = set(sample_nodes) - connected_nodes
	
	return connected_nodes, orphan_nodes

def getDataSorting(connected_nodes, sample_df):
	"""
	Sorts covariate matrix feature order such that connected nodes are first
	followed by orphan nodes and nodes not in interactome
	
	Parameters:
	-----------
	connected_nodes : set,
					  nodes found in subgraph

	sample_df : array-like, shape (n_samples, n_features)
				covariate matrix, prior to data spliting


	Returns:
	--------
	sample_df_sorted : array-like, shape(n_samples, n_features)
					   feature sorted covariate matrix

	ordered_nodelist : array-like
					   list of nodes found in interactome in order of sorting in covariate matrix

	num_to_node : dict
				  dictionary with index of feature (with respect to ordering of covariate matrix)
				  as keys and the names of the nodes as values

	"""

	sample_df_sorted = deepcopy(sample_df)
	sample_df_sorted['IN_INTERACTOME'] = sample_df["node"].isin(list(connected_nodes)).tolist()
	sample_df_sorted = sample_df_sorted.sort_values(by="IN_INTERACTOME", ascending=False).reset_index(drop=True)
	
	#get dictionary to map node to number
	num_to_node = {}
	for i,nod in enumerate(sample_df_sorted['node'].tolist()):
		num_to_node[i] = nod
	
	#get ordered list of nodes in interactome
	ordered_nodelist = sample_df_sorted.loc[sample_df_sorted['IN_INTERACTOME'] == True]['node'].tolist()
	
	#delete 'IN_INTERACTOME' column
	sample_df_sorted = sample_df_sorted.drop(columns = ['IN_INTERACTOME', 'node'])
	
	return sample_df_sorted, ordered_nodelist, num_to_node

def getLaplacian(g, ordered_nodelist, orphan_nodes):
	"""
	Calculates laplacian matrix with respect to ordering of 
	covariate matrix, appends identity matrix for orphan nodes
	
	Parameters:
	-----------
	g : NetworkX graph,
		reduced version

	ordered_nodelist : array-like
					   list of nodes found in interactome in order of sorting in covariate matrix
	
	orphan_nodes : set,
				   nodes not found in subgraph 

	Returns:
	--------
	* : tuple, 
		(Un-normalized Laplacian, normalized laplacian)


	"""
	L_norm = nx.normalized_laplacian_matrix(g, nodelist = ordered_nodelist, weight = 'confidence')
	L = nx.laplacian_matrix(g, nodelist = ordered_nodelist, weight = 'confidence')
	return csr_matrix(scipy.linalg.block_diag(L.todense(),np.eye(len(orphan_nodes)))), \
		csr_matrix(scipy.linalg.block_diag(L_norm.todense(),np.eye(len(orphan_nodes))))


class Preprocessing():

	"""
	Does all preprocessing steps 
	"""
	
	def __init__(self):
		self.g = None
		self.connected_nodes = None
		self.orphan_nodes = None
		self.sorted_X = None
		self.ordered_nodelist = None
		self.num_to_node = None
		self.L = None
		self.L_norm = None
		
	def transform(self,sample_nodes, graph_nodes, interactome, X, save_location, load_graph = False):

		"""
		Parameters:
		-----------
		sample_nodes : array-like, 
				  list of nodes/features found in dataset

		graph_nodes : array-like,
				  list of nodes/features found in graph/interactome

		interactome : Pandas dataframe
				3 column dataframe [node1, node2, confidence] representing graph

		X : array-like, shape (n_samples, n_features)
			covariate matrix, prior to data spliting

		save_location : str
						Path to save reduced graph

		load_graph : bool 
					 If True, load reduced graph from save_location, else generate
					 reduced graph

		"""
		
		if load_graph == False:
			self.g = getReducedGraph(sample_nodes, graph_nodes, interactome)
			self.g.to_csv(save_location, header=None, index=None, sep='\t')
		
		self.g = nx.read_edgelist(save_location, 
					 data=(('confidence',float),))
		
		self.connected_nodes, self.orphan_nodes = \
			getNodeCharacterization(self.g, sample_nodes)
		
		self.sorted_X, self.ordered_nodelist, self.num_to_node = \
			getDataSorting(self.connected_nodes,X)
		
		self.L, self.L_norm = getLaplacian(self.g, self.ordered_nodelist, self.orphan_nodes, )
		