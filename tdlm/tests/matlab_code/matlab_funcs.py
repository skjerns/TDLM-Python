# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 08:27:39 2021

A set of convenience functions that call original matlab functions
a working matlab installation must be present for those and the
python package for matlab must be installed

@author: Simon Kern
"""
import logging

import numpy as np
from scipy.io import loadmat
from sklearn.linear_model._base import LinearModel


def get_matlab_engine(seed=1):
    """
    get a reference to a MATLAB engine.
    If a matlab engine has been called within this session, it is reused.
    """
    import matlab
    if not '_MATLAB_ENGINE_REF' in matlab.__dict__: # this is hacky, I know
        import matlab.engine
        print('Starting MATLAB engine')
        ml = matlab.engine.start_matlab()
        ml.rng(seed) # set seed
        matlab._MATLAB_ENGINE_REF = ml
    return matlab._MATLAB_ENGINE_REF


def compare_mat(mat_file, vardict):
    """
    compare variables stored in a MAT file with python variables

    usage:
        1. save workspace in matlab to workspace.mat
        2. run equivalent function in python
        3. execute function with compare_mat('workspace.mat', locals())
    """
    from mat73 import loadmat as loadmat73
    try:
        data = loadmat73(mat_file)
    except TypeError:
        data = loadmat(mat_file)

    maxlen = max([len(x) for x in list(data)])
    errors = []
    checks = []

    for key, values_mat in data.items():
        msg = ''
        if '__' in key: continue
        msg += f'{key}' + '.' * (maxlen+3-len(key))
        if key not in vardict:
            msg+= 'not found'
            continue
        values_py = vardict[key]
        if isinstance(values_mat, np.ndarray):
            values_py = np.squeeze(values_py)
            values_mat = np.squeeze(values_mat)
            if values_py.dtype==bool or values_mat.dtype==bool:
                values_py = values_py.astype(int)
                values_mat = values_mat.astype(int)

        try:
            np.testing.assert_allclose(values_py, values_mat, rtol=1e-05)
            msg += 'ok'
        except Exception as e:
            try:
                # some vars are actually indices, which start at 1 in matlab
                np.testing.assert_allclose(values_py+1, values_mat)
                msg += 'ok'
            except Exception:
                msg += 'failed'
                errors.append(f'\n{key} is unequal {e}\n-------\n')
        checks.append(msg)



    for err in errors:
        print (err)

    for chk in checks:
        print (chk)

def autoconvert(func):
    """
    a decorator that turns all input to MATLAB datatypes
    and all outputs to Python datatypes
    """
    try:
        import matlab
    except Exception:
        logging.error("Matlab for Python isn't installed in this distribution")
        return None
    matlab.float64 = matlab.double
    matlab.float32 = matlab.double
    matlab.bool = matlab.logical

    def wrapped(*args, **kwargs):
        args = [matlab.__dict__[arg.dtype.name](arg.tolist()) if isinstance(arg, np.ndarray) else arg for arg in args]
        results = func(*args, **kwargs)
        if not isinstance(results, tuple): results = (results,)
        results_np = [np.array(res) if 'mlarray' in str(type(res)) else res for res in results]
        return results_np[0] if len(results_np)==1 else results_np
    return wrapped


class MATLABLasso(LinearModel):

    def __init__(self, Lambda=0.006, Standardize=False, max_iter=100, **kwargs):
        self.Lambda = Lambda
        self.penalty = 'l1'
        self.max_iter = max_iter
        self.Standardize = Standardize
        
    def fit(self, X, y, **kwargs):
        X = np.array(X)
        y = np.array(y)
        classes = np.unique(y)
        binominal = len(classes) == 2
        y = np.reshape(y, [-1, 1])
        if binominal:
            coef, info = lassoglm(X, y.astype(bool), 'binomial', 'Alpha', 1.0, 
                                  'Lambda', self.Lambda,  nargout=2, **kwargs)
            intercept = info['Intercept']
        else:
            coef = []
            intercept = []
            for c in classes:
                y_c = y==c
                coef_c, info = lassoglm(X, y_c, 'binomial', 'Alpha', 1.0, 
                                      'Lambda', self.Lambda,  nargout=2, **kwargs)
                intercept.append(info['Intercept'])
                coef.append(coef_c)
        self.intercept_ = np.squeeze(intercept)
        self.coef_ = np.squeeze(coef)
        self.classes_ = classes
        
    def predict_proba(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
            
    # def predict(self, *args, **kwargs):
        # return self.predict_proba(*args, **kwargs).argmax(-1)

@autoconvert
def lassoglm(data_x, data_y, *args, **kwargs):
    """wrapper for matlab lassoglm"""
    ml = get_matlab_engine()
    return ml.lassoglm(data_x, data_y, *args, **kwargs)

def rand(*args):
    ml = get_matlab_engine()
    func = autoconvert(ml.rand)
    return np.array(func(*args))


def randn(*args, **kwargs):
    ml = get_matlab_engine()
    func = autoconvert(ml.randn)
    return np.array(func(*args, **kwargs)).reshape(*args)


def eig(*args, **kwargs):
    ml = get_matlab_engine()
    func = autoconvert(ml.eig)
    return np.array(func(*args, **kwargs, nargout=2))[::-1]

def randsample(*args, **kwargs):
    ml = get_matlab_engine()
    func = autoconvert(ml.randsample)
    return np.array(func(*args, **kwargs)).reshape([args[1]])

def repmat(a, dims):
    return np.matlib.repmat(a, *dims)

def mvnrnd(*args, **kwargs):
    ml = get_matlab_engine()
    func = autoconvert(ml.mvnrnd)
    return np.array(func(*args, **kwargs))

def randint(*args, **kwargs):
    import matlab
    ml = get_matlab_engine()
    func = autoconvert(ml.randi)
    if len(args)<=2:
        args = (matlab.int64(args),)
    elif len(args)==3:
        args = (args[:2], [args[3]])
    return np.array(func(*args, **kwargs), dtype=int)-1

def gamma(gamA, gamB, **kwargs):
    import matlab
    ml = get_matlab_engine()
    gamA = matlab.double([gamA])
    gamB = matlab.double([gamB])
    return autoconvert(ml.gamrnd)(gamA, gamB, **kwargs)




@autoconvert
def pinv(*args, **kwargs):
    ml = get_matlab_engine()
    return ml.pinv(*args, **kwargs)

@autoconvert
def toeplitz(*args, **kwargs):
    ml = get_matlab_engine()
    return ml.toeplitz(*args, **kwargs)

# @autoconvert
# def rand(*args, **kwargs):
#     ml = get_matlab_engine()
#     return ml.rand(*args, **kwargs)


