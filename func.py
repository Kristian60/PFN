__author__ = 'thoru_000'

from pandas import *
import numpy as np

ass = DataFrame(np.arange(1,101).reshape(10,10))
ref = DataFrame(np.arange(1,11).reshape(10,1))

def getBeta(asset_i,reference):
	reff = Series(reference.iloc[:,0])
	feedback = []
	for asset in range(len(asset_i.columns)):
		asset_col = Series(asset_i.iloc[:,asset])
		cov = float(asset_col.cov(reff))
		varr = float(reference.std(ddof=1)**2)
		feedback.append(cov / varr)
	return feedback

#print getBeta(ass,ref)

