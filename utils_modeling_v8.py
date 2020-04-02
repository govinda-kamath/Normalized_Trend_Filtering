# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, ShuffleSplit, GridSearchCV, cross_val_score, StratifiedKFold, train_test_split
from sklearn.base import clone
from sklearn.metrics import mean_squared_error

from utils_plots_v2 import *

from joblib import Parallel, delayed
import cvxpy as cp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
import seaborn as sns
import itertools
import pycasso

################ PRIVATE METHODS ################

def loss_fn(X, Y, beta, penalty):
	"""
	Get loss function for objective function

	Parameters:
	-----------
	X : CVXPY-variable
		covariate matrix

	Y : CVXPY-variable, 
		responses to samples

	beta : CVXPY-variable
		   beta vector 

	penalty : string
			  Either 'Ridge' or 'Lasso'

	Returns:
	--------
	* : Loss function for optimization
	"""

	if penalty == 'Ridge':
		return cp.pnorm(cp.matmul(X, beta) - Y, p=2)**2

	if penalty == 'Lasso':
		return cp.norm2(cp.matmul(X, beta) - Y)**2


def regularizer(beta, penalty, L):

	"""
	Get regulizaration method for objective function

	
	Parameters:
	-----------
	beta : CVXPY-variable
		   beta vector 

	penalty : string
			  Either 'Ridge' or 'Lasso'

	L : array-like, shape (n_features, n_features)
	    Laplacian matrix

	Returns:
	--------
	* : Regularization method for optimization
	"""

	if penalty == 'Ridge':
		if L is not None:
			return cp.sum(cp.quad_form(beta,L))
		else:
			return cp.pnorm(beta, p=2)**2

	if penalty == 'Lasso':
		if L is not None:
			return cp.norm1(L*beta)
		else:
			return cp.norm1(beta)

def objective_fn(X, Y, beta, alpha, L, penalty):

	"""
	Define objective function for optimization
	
	Parameters:
	-----------
	X : CVXPY-variable
		covariate matrix

	Y : CVXPY-variable, 
		responses to samples

	beta : CVXPY-variable
		   beta vector 

	alpha : float,
			value for alpha, strength of regularization

	L : array-like, shape (n_features, n_features)
	    Laplacian matrix
	
	penalty : string
			  Either 'Ridge' or 'Lasso'
			  
	Returns:
	--------
	* : Objective function
	"""	

	return 0.5/(len(X)) * loss_fn(X, Y, beta, penalty) + alpha * regularizer(beta, penalty, L) # from sklearn lasso


################ PUBLIC METHODS ################

def getOptimization(X, y, alpha_value, L, penalty, solver = cp.SCS, params = {}, verbose = True):

	"""
	Optimizes objective function
	
	Parameters:
	-----------
	X : CVXPY-variable
		covariate matrix

	Y : CVXPY-variable, 
		responses to samples

	alpha_value : float,
			value for alpha, strength of regularization

	L : array-like, shape (n_features, n_features)
	    Laplacian matrix
	
	penalty : string
			  Either 'Ridge' or 'Lasso'

	solver : CVXPY-solver, 
			 optimization method

	params : dictionary, solver parameter : value
			 dictionary to specialize 
			  
	Returns:
	--------
	beta : array-lie, shape (n_features, )
		   optimized beta vector, coefficients for features
	"""	

	beta = cp.Variable(X.shape[1])
	alpha = cp.Parameter(nonneg=True)
	alpha.value = alpha_value
	problem = cp.Problem(cp.Minimize(objective_fn(X, y, beta, alpha, L, penalty)))

	problem.solve(solver = solver, verbose = verbose, **params)
	
	if beta.value is None:
		return np.zeros(X.shape[1])
	else:
		return beta.value


def getFeatures(alpha, beta, threshold, max_features):

	"""
	Gets features with coefficients above threshold to be selected

		Parameters:
	-----------
	alpha : float,
			value for alpha, strength of regularization

	beta : array-like, shape (n_features, )
		   optimized beta vector, coefficients for features
	
	threshold : float
				minimum absolute value coefficients must be to be selected 
				as a feature

	max_features : int
				   max number of features that can be selected
			  
	Returns:
	--------
	* : array-like, shape (num_of_selected_features, )
		array with indicies of selected features with respect to ordering
		of features in X matrix

	"""

	top_cols = np.argsort(-np.abs(beta))[:max_features]
	return np.array([i for i in top_cols if np.abs(beta[i]) > threshold])


def getScoring(regr, X_train, y_train, X_test, y_test, features, alpha):

	"""
	Fit linear regression after feature selection step and calculate train/test mean squared error
	
	Parameters:
	-----------
	regr : Sklearn regressor template, LinearRegression()
		   Regression template to build model

	X_train : array-like, shape (n_samples, n_features)
			  Covariate matrix with train data

	y_train : array-like, shape (n_samples,)
			  Response vector with train data

	X_test : array-like, shape (n_samples, n_features)
			 Covariate matrix with test data

	y_test : array-like, shape (n_samples,)
			 Response vector with test data

	features : array-like, shape (n_selected_features,)
			   List of indicies of features selected

	alpha : float,
			value for alpha, strength of regularization
			  
	Returns:
	--------
	* : tuple,
		Mean squared error (MSE) from train set and MSE from test set

	"""
	
	if len(features) == 0:
		mse_train = mean_squared_error(y_train, np.mean(y_train) * np.ones(len(y_train)))
		mse_test = mean_squared_error(y_test, np.mean(y_train) * np.ones(len(y_test)))
		return (mse_train, mse_test)
		
	else:
		regr = clone(regr)
		X_train = X_train[:,features]
		X_test = X_test[:,features]
		regr.fit(X_train,y_train)
		mse_train = mean_squared_error(y_train, regr.predict(X_train))
		mse_test = mean_squared_error(y_test, regr.predict(X_test))
		return (mse_train, mse_test)

def getBestParam(gridsearch_results_raw, force_features):
	
	"""
	Analyzes gridsearch to get best alpha and threshold
	
	Parameters:
	-----------
	gridsearch_results_raw : pandas dataframe
							 Results from gridsearch with columns ['alpha', 'threshold','beta', 'features', 'Train MSE', 'Test MSE']


	force_features : bool,
					 if True, the hyperparameters chosen must select for at least 1 feature
			  
	Returns:
	--------
	* : tuple,
		Processed gridsearch, best alpha value, best threshold value
	"""

	gridsearch_results_raw['num_features'] = [len(row) for row in gridsearch_results_raw.features]
	gridsearch_results = gridsearch_results_raw[['alpha', 'threshold', 'num_features', 'Train MSE', 'Test MSE']]

	if force_features:
		best_idx = gridsearch_results.sort_values(by = 'Test MSE').loc[gridsearch_results['num_features'] >= 1].index[0]
		return (gridsearch_results, \
			gridsearch_results.iloc[best_idx]['alpha'], gridsearch_results.loc[best_idx]['threshold'])
	else:
		best_idx = gridsearch_results['Test MSE'].idxmin()
		return (gridsearch_results, \
			gridsearch_results.iloc[best_idx]['alpha'], gridsearch_results.loc[best_idx]['threshold'])


def getRegressionTemplate(penalty):
	"""
	Model template for feature selection step in SklearnRegression() class
	
	Parameters:
	-----------
	penalty : string
			  Choose between Sklearn's Lasso or Ridge
			  
	Returns:
	--------
	* : Sklearn regressor
		Ridge() or Lasso()
	"""
	if penalty == 'Ridge':
		return Ridge()

	if penalty == 'Lasso':
		return Lasso()


class LaplacianRegression():

	"""
	Pipeline for feature selection using Laplacian
	"""
	
	def __init__(self, penalty):

		"""
		Parameters:
		-----------
		penalty : string,
				  Ridge or Lasso

		"""

		self.gridsearch_results_raw = None
		self.gridsearch_results = None
		self.alpha_ = None
		self.beta_ = None
		self.feats_ = None
		self.regr_ = None
		self.threshold_ = None
		self.param_list = None
		self.y_avg = None
		self.max_features = None
		self.y_predict = None
		self.threshold_list = None
		self.alpha_list = None
		self.penalty = penalty
		self.coef_ = None
		self.temp = None


	def fit(self, X_train, y_train, X_test, y_test, alpha_list, threshold_list, L = None, max_features = 10,
			solver = cp.SCS, solver_params = {}, force_features = False, verbose = True, n_jobs = -1):
		
		"""
		Hyperparameter/feature selection and fitting final model

		Parameters:
		-----------
		X_train : array-like, shape (n_samples, n_features)
			  Covariate matrix with train data

		y_train : array-like, shape (n_samples,)
				  Response vector with train data

		X_test : array-like, shape (n_samples, n_features)
				 Covariate matrix with test data

		y_test : array-like, shape (n_samples,)
				 Response vector with test data

		alpha_list : array-like,
					 list of alphas for gridsearch

		threshold_list : array-like
						 list of thresholds (min abs value for coefficients to be selected as feature) 
						 for gridsearch

		L : array-like, shape (n_features, n_features)
	    	Laplacian matrix

		max_features : int
				   max number of features that can be selected

		solver : CVXPY-solver, 
			 	 optimization method

		solver_params : dictionary, solver parameter : value
			 	 dictionary to specialize 

		force_features : bool,
					 if True, the hyperparameters chosen must select for at least 1 feature

		verbose : int or bool
				  Joblib's parallel verbose  

		n_jobs : int
				 Joblib's number of workers, for gridsearch
		"""

		param_list = list(itertools.product(alpha_list, threshold_list))
		self.param_list = param_list
		self.max_features = max_features
		self.threshold_list = threshold_list
		self.alpha_list = alpha_list
		
		regr = LinearRegression()
		self.temp = Parallel(n_jobs=n_jobs, verbose = verbose)(delayed(self._automated_gridsearch)(
				X_train = deepcopy(X_train),
				y_train = deepcopy(y_train),
				X_test = deepcopy(X_test),
				y_test = deepcopy(y_test),
				alpha = alph,
				regr = clone(regr), 
				threshold_list = self.threshold_list,
				max_features = self.max_features,
				L = L, 
				solver = solver, 
				solver_params = solver_params,
				penalty = self.penalty,
				verbose = verbose) for alph in self.alpha_list), 

		self.gridsearch_results_raw = []
		for i in np.squeeze(self.temp):
			for j in i:
				self.gridsearch_results_raw.append(tuple(j))

		self.gridsearch_results_raw = pd.DataFrame(self.gridsearch_results_raw, 
			columns = ['alpha', 'threshold','beta', 'features', 'Train MSE', 'Test MSE'])
		
		self.gridsearch_results, self.alpha_, self.threshold_ = getBestParam(self.gridsearch_results_raw, force_features)

		self.y_avg = np.mean(y_train)
		
		self.beta_ = getOptimization(X_train, y_train, self.alpha_, L = L, 
									 solver = solver, params = solver_params, penalty = self.penalty, verbose = verbose)
		self.feats_ = getFeatures(self.alpha_, self.beta_, threshold = self.threshold_, max_features = self.max_features)
		
		if len(self.feats_) == 0:
			self.regr_ = 'DID NOT CHOOSE ANY FEATURES'
		else:    
			self.regr_ = clone(regr)
			self.regr_.fit(X_train[:,self.feats_],y_train)        
			self.coef_ = self.regr_.coef_
	
	def predict(self, X_test, y_test):

		"""
		Predict response (raw predictions in self.y_predict) and calculates MSE

		Parameters:
		-----------
		X_test : array-like, shape (n_samples, n_features)
				 Covariate matrix with test data

		y_test : array-like, shape (n_samples,)
				 Response vector with test data

		Returns:
		--------
		* : Mean squared error between y_test and model predictions
		"""

		if len(self.feats_) == 0:
			self.y_predict = self.y_avg*np.ones(len(X_test))
			return mean_squared_error(y_test, self.y_predict)

		else:
			self.y_predict = self.regr_.predict(X_test[:,self.feats_])
			return mean_squared_error(y_test, self.y_predict)
   

	def _automated_gridsearch(self, X_train, y_train, X_test, y_test, alpha, regr, 
							  threshold_list, max_features, solver, solver_params, L, penalty, verbose):

		"""
		Hyperparameter/feature selection and fitting final model

		Parameters:
		-----------
		X_train : array-like, shape (n_samples, n_features)
			  Covariate matrix with train data

		y_train : array-like, shape (n_samples,)
				  Response vector with train data

		X_test : array-like, shape (n_samples, n_features)
				 Covariate matrix with test data

		y_test : array-like, shape (n_samples,)
				 Response vector with test data

		alpha : float,
			value for alpha, strength of regularization

		regr : Sklearn regressor template, LinearRegression()
		       Regression template to build model

		threshold_list : array-like
						 list of thresholds (min abs value for coefficients to be selected as feature) 
						 for gridsearch

		max_features : int
				   max number of features that can be selected

		solver : CVXPY-solver, 
			 	 optimization method

		solver_params : dictionary, solver parameter : value
			 	 dictionary to specialize 

		L : array-like, shape (n_features, n_features)
	    	Laplacian matrix

	    penalty : string
			  	  Either 'Ridge' or 'Lasso'

		verbose : int or bool
				  CVXPY's solver verbose  

		"""
		
		regr = clone(regr)
		
		beta = getOptimization(X_train, y_train, alpha, L = L, 
							   solver = solver, params = solver_params, penalty = penalty, verbose = verbose)
		retVal = []
		for th in threshold_list:
			feats = getFeatures(alpha, beta, th, max_features)
			score = getScoring(regr, X_train, y_train, 
						   X_test, y_test, feats, alpha)
			retVal.append((alpha, th, beta, feats, score[0], score[1]))
		return retVal

	def plotgridsearch(self, save_location = None):
		"""
		Plot gridsearch 

		Parameters:
		-----------

		save_location : string
						Path to save gridsearch plot
		"""
		getGridsearchPlot(self.gridsearch_results, self.alpha_list, self.threshold_list, save_location)



class NonConvexRegression():

	'''
	Pipeline for nonconvex regression, similar to LaplacianRegression() class
	'''
	
	def __init__(self, penalty):

		"""
		Parameters:
		-----------
		penalty : string,
				  Ridge or Lasso

		"""

		self.alpha_list = None
		self.threshold_list = None
		self.max_features = None
		self.gridsearch_regression = None
		self.gridsearch_results_raw = []
		self.gridsearch_results = None
		self.alpha_ = None
		self.threshold_ = None
		self.y_avg = None
		self.beta_ = None
		self.feats_ = None
		self.regr_ = None
		self.y_predict = None
		self.penalty = penalty
		
	def fit(self, X_train, y_train, X_test, y_test, alpha_list, threshold_list, max_features, force_features = True):
		
		"""
		Hyperparameter/feature selection and fitting final model

		Parameters:
		-----------
		X_train : array-like, shape (n_samples, n_features)
			  Covariate matrix with train data

		y_train : array-like, shape (n_samples,)
				  Response vector with train data

		X_test : array-like, shape (n_samples, n_features)
				 Covariate matrix with test data

		y_test : array-like, shape (n_samples,)
				 Response vector with test data

		alpha_list : array-like,
					 list of alphas for gridsearch

		threshold_list : array-like
						 list of thresholds (min abs value for coefficients to be selected as feature) 
						 for gridsearch

		max_features : int
				   max number of features that can be selected

		force_features : bool,
					 if True, the hyperparameters chosen must select for at least 1 feature

		"""
		self.alpha_list = alpha_list
		self.threshold_list = threshold_list
		self.max_features = max_features
		
		
		regr = LinearRegression()
		
		s = pycasso.Solver(X_train, y_train, lambdas=alpha_list, penalty = self.penalty)
		s.train()
		
		self.gridsearch_regression = s
		
		for i in range(len(self.alpha_list)):
			for j in range(len(self.threshold_list)):
				alpha = alpha_list[i]
				beta = s.coef()['beta'][i]
				threshold = self.threshold_list[j]
				feats = getFeatures(alpha, beta, threshold, self.max_features)
				
				score = getScoring(regr, X_train, y_train, X_test, y_test, feats, alpha)
				self.gridsearch_results_raw.append([alpha, threshold, beta, feats, score[0], score[1]])
		self.gridsearch_results_raw = pd.DataFrame(self.gridsearch_results_raw, 
							columns = ['alpha', 'threshold','beta', 'features', 'Train MSE', 'Test MSE'])
		
		self.gridsearch_results, self.alpha_, self.threshold_ = getBestParam(self.gridsearch_results_raw, force_features)
		
		self.y_avg = np.mean(y_train)
		
		#grab best beta
		if force_features:
			best_idx = self.gridsearch_results.sort_values(by = 'Test MSE').loc[self.gridsearch_results['num_features'] >= 1].index[0]
		else:
			best_idx = self.gridsearch_results['Test MSE'].idxmin()
		
		self.beta_ = self.gridsearch_results_raw.iloc[best_idx]['beta']
		
		self.feats_ = getFeatures(self.alpha_, self.beta_, threshold = self.threshold_, max_features = self.max_features)
		
		if len(self.feats_) == 0:
			self.regr_ = 'DID NOT CHOOSE ANY FEATURES'
		else:    
			self.regr_ = clone(regr)
			self.regr_.fit(X_train[:,self.feats_],y_train)        
			self.coef_ = self.regr_.coef_

			
	def predict(self, X_test, y_test):

		"""
		Predict response (raw predictions in self.y_predict) and calculates MSE

		Parameters:
		-----------
		X_test : array-like, shape (n_samples, n_features)
				 Covariate matrix with test data

		y_test : array-like, shape (n_samples,)
				 Response vector with test data

		Returns:
		--------
		* : Mean squared error between y_test and model predictions
		"""
		
		if len(self.feats_) == 0:
			self.y_predict = self.y_avg*np.ones(len(X_test))
			return mean_squared_error(y_test, self.y_predict)

		else:
			self.y_predict = self.regr_.predict(X_test[:,self.feats_])
			return mean_squared_error(y_test, self.y_predict)
		
	def plotgridsearch(self, save_location = None):

		"""
		Plot gridsearch 

		Parameters:
		-----------

		save_location : string
						Path to save gridsearch plot
		"""
		getGridsearchPlot(self.gridsearch_results, self.alpha_list, self.threshold_list, save_location)



class SklearnRegression():

	'''
	Sanity check to make sure results from LaplacianRegression class (when L = None) are consistent with Sklearn
	'''

	def __init__(self, penalty):

		"""
		Parameters:
		-----------
		penalty : string,
				  Ridge or Lasso

		"""

		self.gridsearch_results_raw = None
		self.gridsearch_results = None
		self.alpha_ = None
		self.beta_ = None
		self.feats_ = None
		self.regr_ = None
		self.threshold_ = None
		self.param_list = None
		self.y_avg = None
		self.max_features = None
		self.y_predict = None
		self.threshold_list = None
		self.alpha_list = None
		self.penalty = penalty
		self.coef_ = None
		self.temp = None

	def fit(self, X_train, y_train, X_test, y_test, alpha_list, threshold_list, max_features = 10,
			force_features = False, verbose = True, n_jobs = -1):
		
		"""
		Hyperparameter/feature selection and fitting final model

		Parameters:
		-----------
		X_train : array-like, shape (n_samples, n_features)
			  Covariate matrix with train data

		y_train : array-like, shape (n_samples,)
				  Response vector with train data

		X_test : array-like, shape (n_samples, n_features)
				 Covariate matrix with test data

		y_test : array-like, shape (n_samples,)
				 Response vector with test data

		alpha_list : array-like,
					 list of alphas for gridsearch

		threshold_list : array-like
						 list of thresholds (min abs value for coefficients to be selected as feature) 
						 for gridsearch

		max_features : int
				   max number of features that can be selected

		force_features : bool,
					 if True, the hyperparameters chosen must select for at least 1 feature

		verbose : int or bool
				  Joblib's parallel verbose  

		n_jobs : int
				 Joblib's number of workers, for gridsearch
		"""
		param_list = list(itertools.product(alpha_list, threshold_list))
		self.param_list = param_list
		self.max_features = max_features
		self.threshold_list = threshold_list
		self.alpha_list = alpha_list
		
		regr = LinearRegression()
		opt_temp = getRegressionTemplate(self.penalty)

		temp = Parallel(n_jobs=n_jobs, verbose = verbose)(delayed(self._automated_gridsearch)(
				X_train = deepcopy(X_train),
				y_train = deepcopy(y_train),
				X_test = deepcopy(X_test),
				y_test = deepcopy(y_test),
				alpha = alph,
				regr = clone(regr),
				template = clone(opt_temp),
				threshold_list = self.threshold_list,
				max_features = self.max_features,) for alph in self.alpha_list)

		self.gridsearch_results_raw = []
		for i in np.squeeze(temp):
			for j in i:
				self.gridsearch_results_raw.append(tuple(j))

		self.gridsearch_results_raw = pd.DataFrame(self.gridsearch_results_raw, 
			columns = ['alpha', 'threshold','beta', 'features', 'Train MSE', 'Test MSE'])
		
		self.gridsearch_results, self.alpha_, self.threshold_ = getBestParam(self.gridsearch_results_raw, force_features)

		self.y_avg = np.mean(y_train)
		
		final_template = clone(opt_temp)

		final_template.set_params(**{'alpha' : self.alpha_})
		final_template.fit(X_train, y_train)

		self.beta_ = final_template.coef_

		self.feats_ = getFeatures(self.alpha_, self.beta_, threshold = self.threshold_, max_features = self.max_features)
		
		if len(self.feats_) == 0:
			self.regr_ = 'DID NOT CHOOSE ANY FEATURES'
		else:    
			self.regr_ = clone(regr)
			self.regr_.fit(X_train[:,self.feats_],y_train)        
			self.coef_ = self.regr_.coef_

	def predict(self, X_test, y_test):

		"""
		Predict response (raw predictions in self.y_predict) and calculates MSE

		Parameters:
		-----------
		X_test : array-like, shape (n_samples, n_features)
				 Covariate matrix with test data

		y_test : array-like, shape (n_samples,)
				 Response vector with test data

		Returns:
		--------
		* : Mean squared error between y_test and model predictions
		"""

		if len(self.feats_) == 0:
			self.y_predict = self.y_avg*np.ones(len(X_test))
			return mean_squared_error(y_test, self.y_predict)

		else:
			self.y_predict = self.regr_.predict(X_test[:,self.feats_])
			return mean_squared_error(y_test, self.y_predict)
   
	def _automated_gridsearch(self, X_train, y_train, X_test, y_test, alpha, regr, template,
							  threshold_list, max_features):

		"""
		Hyperparameter/feature selection and fitting final model

		Parameters:
		-----------
		X_train : array-like, shape (n_samples, n_features)
			  Covariate matrix with train data

		y_train : array-like, shape (n_samples,)
				  Response vector with train data

		X_test : array-like, shape (n_samples, n_features)
				 Covariate matrix with test data

		y_test : array-like, shape (n_samples,)
				 Response vector with test data

		alpha : float,
			value for alpha, strength of regularization

		regr : Sklearn regressor template, LinearRegression()
		       Regression template for building final models

		template : Sklearn regressor template for feature selection step, Lasso() or Ridge()
				   Regression template for feature selection

		threshold_list : array-like
						 list of thresholds (min abs value for coefficients to be selected as feature) 
						 for gridsearch

		max_features : int
				   max number of features that can be selected

		"""
		
		template.set_params(**{'alpha' : alpha})
		template.fit(X_train, y_train)
		beta = template.coef_
		
		retVal = []

		for th in threshold_list:
			feats = getFeatures(alpha, beta, th, max_features)
			score = getScoring(regr, X_train, y_train, 
						   X_test, y_test, feats, alpha)
			retVal.append((alpha, th, beta, feats, score[0], score[1]))
		return retVal

	def plotgridsearch(self, save_location = None):

		"""
		Plot gridsearch 

		Parameters:
		-----------

		save_location : string
						Path to save gridsearch plot
		"""

		getGridsearchPlot(self.gridsearch_results, self.alpha_list, self.threshold_list, save_location)


