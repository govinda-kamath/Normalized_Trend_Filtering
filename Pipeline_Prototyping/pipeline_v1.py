##### IMPORT MODULES #####
import pandas as pd
import os
from tqdm import tqdm_notebook, tnrange
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

import scipy
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, vstack, hstack

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import cvxpy as cp

import pickle

from joblib import Parallel, delayed

module_path = os.path.abspath(os.path.join('..'))

%matplotlib inline


##### PIPELINE #####

class LinearRegression:

    def __init__ (self,):
        self.intersection_proteins = set()
        self.connected_proteins = set()
        self.orphan_proteins = set()
        self.graph = None
        self.ordered_protein_names = []
        self.ordered_in_interactome_bool = []
        self.ordered_protein_names_connected = []
        self.L = None
        self.A = None
        self.selected_features = []
        self.regr = None
        self.alpha = None
        self.split_num = None
        #delete
        self.X_train = None
        self.reg = None
        
    def preliminary_analysis(self,path):
        self.graph = self.getGraph(path)
        print('SHOW GRAPHS')
        

    def fit(self, X_train, y_train, data_protein, interactome_protein, alpha, split_num, coef_threshold = 0.001, 
            max_features = 10, method = 'vanilla', preset = None):   
        '''
        Parameters:
        -----------
        
        X_train: see example file 
        
        y_train: array-like, list of responses 
        
        data_protein : array-like, list of proteins/nodes found in train data
        
        interactome_protein : array-like, list of proteins/nodes found in interactome 
        
        alpha : float, for regularization strength
        
        split_num : int, to serialize the split if using cross validation (can ignore for now and just put in any number)
        
        coef_threshold : float, threshold for feature coeffiencient in order to be selected
        
        max_features : int, max number of features selected
        
        method : str, 'vanilla' or 'laplacian'
        
        preset : if you have a laplacian already calculated... 
        
        Returns:
        --------
        
        
        '''
        
        
        #save signature
        self.alpha = alpha
        self.split_num = split_num
        
        self.data_protein = data_protein
        self.interactome_protein = interactome_protein
        self.graph = self.getGraph(path)
        
        self.preprocessing(self.data_protein, self.interactome_protein)
        X = self.getSortData(X_train, self.connected_proteins)
        X = X.values
        self.X_train = deepcopy(X)
        
        if method == 'vanilla':
            reg = self.getVanilla(self.orphan_proteins, self.connected_proteins,)
            
        if method == 'laplacian':
            if preset is not None:
                L, reg = preset
            else:
                L,reg = self.getLaplacian(self, self.graph, self.ordered_protein_names_connected)
        
        self.reg = reg
        
        self.A = self.getOptimization(self.orphan_proteins, self.connected_proteins, self.X_train, y_train, reg, alpha)
        
        np.save('A_' + str(split_num) + '_' + str(alpha) + '.npy', self.A)
        
        sorted_features = np.argsort(-abs(self.A))[:max_features]
        
        self.selected_features.append([i for i in sorted_features if np.abs(self.A[i])> coef_threshold])
        
        if not self.selected_features[0]:
            pass
        else:
            self.regr = self.getFittedRegression(self.X_train[:,self.selected_features[0]], y_train)
        
        
    
    def predict(self, X_test, y_test):
        if not self.selected_features[0]:
            y_pred = np.mean(y_test) * np.ones(y_test.shape)
        else:
            X_test = self.getSortData(X_test, self.connected_proteins).values
            X_test = X_test[:,self.selected_features[0]] #FLAG
            y_pred = self.regr.predict(X_test)
        return y_pred
        
        
    def preprocessing(self, data_protein, interactome_protein):
        self.intersection_proteins = self.getUnion(self.data_protein, self.interactome_protein)
        self.connected_proteins, self.orphan_proteins = self.getConnectedProteins(self.graph, self.data_protein)
                
        
    def getUnion(self, data_protein, interactome_protein):
        '''
        Gets the union of proteins in dataset and interactome
        '''
        intersection_proteins = set()
        for protein in tqdm_notebook(data_protein):
            if protein in interactome_protein:
                intersection_proteins.add(protein)
    
        return intersection_proteins
    
    def getGraph(self, path):
        g = nx.read_edgelist(path, data=(('confidence',float),))
        return g
    
    def getConnectedProteins(self, g, data_protein,):
        connected_proteins = set(g.nodes())
        orphan_proteins = set(data_protein) - connected_proteins
        return connected_proteins, orphan_proteins
    
    def getSortData(self, data, connected_proteins):
        data['In_Interactome'] = list(data["Gene_Names"].isin(list(connected_proteins)))
        data = data.sort_values(by = 'In_Interactome', ascending = False).reset_index(drop = True)
        self.ordered_protein_names = data['Gene_Names'].tolist()
        self.ordered_protein_names_connected = data[data['In_Interactome'] == True]['Gene_Names'].tolist()
        self.ordered_in_interactome_bool = data['In_Interactome']
        
        data = data.drop(columns = ['In_Interactome', 'Gene_Names'])
        
        data = data.T
        return data
    
    def getLaplacian(self, g, connected_nodelist, orphan_proteins):
        L = nx.normalized_laplacian_matrix(g,connected_nodelist, weight="confidence")
        Reg1 = csr_matrix(scipy.linalg.block_diag(L.todense(),np.eye(len(orphan_proteins))))
        return L, Reg1
        
    def getVanilla(self, orphan_proteins, connected_proteins):
        Reg0 = np.eye(len(orphan_proteins)+len(connected_proteins))
        return Reg0
    
    def getOptimization(self, orphan_proteins, connected_proteins, x, y, reg, alpha):
        A = cp.Variable(len(connected_proteins)+len(orphan_proteins))
        a = cp.Parameter(nonneg=True)
        a.value = alpha
        objective = cp.Minimize((1. / (2*x.shape[0])) * cp.sum_squares(x*A - y)+ a* cp.norm1(reg*A))
        prob = cp.Problem(objective)
        prob.solve()
        return A.value
    
    def getFittedRegression(self, X_train, y_train):
        regr = linear_model.LinearRegression()
        regr.fit(X_train, y_train)
        return regr