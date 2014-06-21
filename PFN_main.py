from __future__ import division
__author__ = 'thoru_000'

#-------------------------------------------------------------------------
#   Preamble
#-------------------------------------------------------------------------
import ystockquote
import numpy as np
import scipy.stats as st
import pandas as pd
from datetime import *

from benchmark import *
import xlrd as xlrd
import locale
from func import *
import plot as plot
import pylab as pl

locale.setlocale(locale.LC_ALL, 'usa_USA')
#-------------------------------------------------------------------------
#   Generating Arrays, and Date-points
#-------------------------------------------------------------------------

assets = np.array([], dtype="a25")
depot = np.array([], dtype=np.int8)
dates = np.array([], dtype=np.datetime64)
region = np.array([], dtype="a25")
sektor = np.array([], dtype="a25")
displayname = np.array([])
temp = []
tdate = []
fx = ["DKK=x", "EURDKK=x", "SEKDKK=x", "HKDDKK=x", "NOKDKK=x"]

#-------------------------------------------------------------------------
#	Load depot and historic prices
#-------------------------------------------------------------------------

depot_wb = xlrd.open_workbook('data\depot.xls')
ws = depot_wb.sheet_by_name('d')
num_rows = ws.nrows

for i in range(1, num_rows):
	assets = np.append(assets, str(ws.cell_value(i, 0)))
	depot = np.append(depot, ws.cell_value(i, 1))
	region = np.append(region, str(ws.cell_value(i, 2)))
	sektor = np.append(sektor, str(ws.cell_value(i, 3)))
	displayname = np.append(displayname, str(ws.cell_value(i, 4)))

PFNpath = "C:/Users/thoru_000/Dropbox/Pers/PFN"
Ppath = "C:/Users/thoru_000/Dropbox/Pers/PFN/prices"

###
### DEV MODE
###

# assets = assets[:3]

###

nAssets = len(assets)

#-------------------------------------------------------------------------
#  get FX prices
#-------------------------------------------------------------------------
for i in range(len(fx)):
	fx[i] = ystockquote.get_price(fx[i])

relevant_fx = [0] * len(assets)

for i in range(len(assets)):
	place = assets[i].find(".")
	if place != -1:
		sintex = assets[i][place + 1:]
		if sintex == "co":
			relevant_fx[i] = 1

		if sintex == "hk":
			relevant_fx[i] = float(fx[3])

		if sintex == "st":
			relevant_fx[i] = float(fx[2])

		if sintex == "de":
			relevant_fx[i] = float(fx[1])
	else:
		relevant_fx[i] = float(fx[0])

#-------------------------------------------------------------------------
# #	Import stockprices, and transform to pricematrix
#-------------------------------------------------------------------------


tDelta = 800
sDate = datetime.today().date() + timedelta(days=-tDelta)
sDate = sDate.strftime("%Y-%m-%d")
eDate = datetime.today().date()
eDate = eDate.strftime("%Y-%m-%d")

print "Downloading " + str(len(assets) * tDelta) + " prices for " + str(len(assets)) + " assets"

for y in assets:
	print "Getting prices for " + str(y)
	temp.append(ystockquote.get_historical_prices(y, sDate, eDate))
print "Download finished"

for y in temp[0]:
	tdate.append(y)

tdate = sorted(tdate, reverse=True)
prices = [[0 for x in xrange(len(assets))] for x in xrange(len(tdate))]

for datee in range(len(tdate)):
	for assetnum in range(nAssets):
		try:
			q = temp[assetnum][tdate[datee]]['Close']
			prices[datee][assetnum] = q
		except KeyError:
			prices[datee][assetnum] = prices[datee - 1][assetnum]

benchmark = getBench(tDelta, "URTH")

size = min(len(prices), 500)

prices = prices[:size]
tdate = tdate[:size]
benchmark = benchmark[:size]

pd_p = pd.DataFrame(prices, index=tdate, columns=assets, dtype=float)
pd_bench = pd.DataFrame(benchmark, index=tdate, columns=["msci"], dtype=float)

print "Calculating returns, covar-matrix, ect..."

#-------------------------------------------------------------------------
# #	Calculate returns, covar-matrix, geo-means and covarmatrix
#-------------------------------------------------------------------------

p = -st.norm.ppf(0.05)

pd_r = (pd_p - pd_p.shift(-1)) / pd_p.shift(-1)
bench_r = (pd_bench - pd_bench.shift(-1)) / pd_bench.shift(-1)
cov = pd_r.cov()
stdev = pd_r.std(ddof=1)
means = pd_r.mean()
VaR = (means - (p * stdev))
beta = getBeta(pd_r, bench_r)

NaV = pd_p * depot * relevant_fx


#
#	Write disp_list
#

vYOY = 180
vMOM = 20
pMat = np.zeros((nAssets,),
                dtype=[("disname", 'a20'), ("curprice", 'f8'), ("Holding", 'f8'), ("NAV", 'f8'), ("x", "f8"),
                       ("y", "f8"), ("r", 'a20'), ("e", 'a20'), ("w", 'a20'), ("q", 'f8')])

for nas in range(nAssets):
	pMat[nas][0] = displayname[nas]  # Asset name
	pMat[nas][1] = prices[0][nas]  # Current price
	pMat[nas][2] = depot[nas]  # Depot holding
	pMat[nas][3] = float(float(pMat[nas][1]) * float(depot[nas]) * float(relevant_fx[nas]))  # Net Asset Value
	pMat[nas][4] = '%.2f' % float(((float(prices[0][nas]) / float(prices[vYOY][nas]) - 1) * 100))  # Should be 180 days
	pMat[nas][5] = '%.2f' % float(((float(prices[0][nas]) / float(prices[vMOM][nas]) - 1) * 100))  # Should be 20 days
	pMat[nas][6] = r'0.00\%'  # Dividend yield
	pMat[nas][7] = sektor[nas]  # Sector
	pMat[nas][8] = region[nas]  # region
	pMat[nas][9] = '%.2f' % float(beta[nas])


#
# Write Winners/Losers
#

wlMat = np.zeros((nAssets,),
                 dtype=[("disname", 'a20'), ("relmove", 'f8'), ("absmove", 'f8')])

for nas in range(nAssets):
	print str(displayname[nas])
	wlMat[nas][0] = displayname[nas]  # Asset name
	wlMat[nas][1] = '%.2f' % float(((float(prices[0][nas]) / float(prices[vMOM][nas]) - 1) * 100))
	print "NAV", str(pMat[nas][3])
	print "MOM", str(pMat[nas][5]/100)
	wlMat[nas][2] = float(float(pMat[nas][3]) * (1 - (1 / (1 + (float(pMat[nas][5])/100)))))
	print "Change", str(wlMat[nas][2])

wlMat = np.sort(wlMat, order="relmove")
h = open('latex/relmove.tex', 'w+')
h.write(r"\begin{tabular}{lr}" + "\r\n")
h.write(r"\toprule" + "\r\n")
h.write(r"Monthly percentage movers" + r'\\' + '\n')
h.write(r"\midrule" + "\r\n")
h.write(wlMat[nAssets - 1][0] + r' & ' + locale.format("%.2f", float(wlMat[nAssets - 1][1]),
                                                       grouping=True) + r'\%\\' + '\n')
h.write(wlMat[nAssets - 2][0] + r' & ' + locale.format("%.2f", float(wlMat[nAssets - 2][1]),
                                                       grouping=True) + r'\%\\' + '\n')
h.write(wlMat[nAssets - 3][0] + r' & ' + locale.format("%.2f", float(wlMat[nAssets - 3][1]),
                                                       grouping=True) + r'\%\\' + '\n')
h.write(r"\midrule" + "\r\n")
h.write(wlMat[2][0] + r' & ' + locale.format("%.2f", float(wlMat[2][1]), grouping=True) + r'\%\\' + '\n')
h.write(wlMat[1][0] + r' & ' + locale.format("%.2f", float(wlMat[1][1]), grouping=True) + r'\%\\' + '\n')
h.write(wlMat[0][0] + r' & ' + locale.format("%.2f", float(wlMat[0][1]), grouping=True) + r'\%\\' + '\n')
h.write(r"\bottomrule" + "\r\n")
h.write(r"\end{tabular}")
h.close()

wlMat = np.sort(wlMat, order="absmove")
g = open('latex/absmove.tex', 'w+')
g.write(r"\begin{tabular}{lr}" + "\r\n")
g.write(r"\toprule" + "\r\n")
g.write(r"Monthly value movers" + r'\\' + '\n')
g.write(r"\midrule" + "\r\n")
g.write(wlMat[nAssets - 1][0] + r' & ' + locale.format("%d", (wlMat[nAssets - 1][2]), grouping=True) + r'\\' + '\n')
g.write(wlMat[nAssets - 2][0] + r' & ' + locale.format("%d", (wlMat[nAssets - 2][2]), grouping=True) + r'\\' + '\n')
g.write(wlMat[nAssets - 3][0] + r' & ' + locale.format("%d", (wlMat[nAssets - 3][2]), grouping=True) + r'\\' + '\n')
g.write(r"\midrule" + "\r\n")
g.write(wlMat[2][0] + r' & ' + locale.format("%d", (wlMat[2][2]), grouping=True) + r'\\' + '\n')
g.write(wlMat[1][0] + r' & ' + locale.format("%d", (wlMat[1][2]), grouping=True) + r'\\' + '\n')
g.write(wlMat[0][0] + r' & ' + locale.format("%d", (wlMat[0][2]), grouping=True) + r'\\' + '\n')
g.write(r"\bottomrule" + "\r\n")
g.write(r"\end{tabular}")
g.close()


pMat = np.sort(pMat, order="NAV")
f = open('latex\dist_list_py.tex', 'w+')
f.write(r"\begin{tabular}{lllrrrrrrr}" + "\r\n")
f.write(r"\toprule" + "\r\n")
f.write(r"Asset & Sector & Region & Price & Holding & NaV & YoY\% & MoM\% & Div. Yield & Beta \\" + "\r\n")
f.write(r"\midrule" + "\r\n")
for nas in range(nAssets - 1, -1, -1):
	f.write(
		pMat[nas][0] + r' & ' + str(pMat[nas][7]) + r' & ' + str(pMat[nas][8]) + r' & ' + str(
			pMat[nas][1]) + r' & ' + locale.format("%d", pMat[nas][2], grouping=True) + r' & \bf{' + locale.format("%d",
		                                                                                                           pMat[
			                                                                                                           nas][
			                                                                                                           3],
		                                                                                                           grouping=True) + r'} & ' + str(
			pMat[nas][4]) + r' & ' + str(pMat[nas][5]) + r'\% & ' + str(pMat[nas][6]) + r' & ' + str(
			pMat[nas][9]) + r'\\' + '\n')
f.write(r"\bottomrule" + "\r\n")
f.write(r"\end{tabular}")
f.close()

#
#	Generate graphs
#

r_EU, r_A, r_US = 0, 0, 0
for entry in range(len(region)):
	if region[entry] == "EU":
		r_EU += float(NaV.iloc[0, entry])
	if region[entry] == "A":
		r_A += float(NaV.iloc[0, entry])
	if region[entry] == "US":
		r_US += float(NaV.iloc[0, entry])

region_fracs = [r_EU / sum(NaV.iloc[0, :]), r_A / sum(NaV.iloc[0, :]), r_US / sum(NaV.iloc[0, :])]
region_lables = ["EU", "AS", "US"]
plot.piechart(region_lables, region_fracs)


# print NaV.iloc[0,:]
"""
x = tdate
fig = plt.figure(1, figsize=(5, 2))
plt.plot(x,prices[1], color='black')
plt.fill_between(x,prices[1],interpolate=True,color="0.95")
plt.grid(False)
plt.rc('font', family='sans-serif', size=8)
plt.ylim( (0, np.max(prices[1])*1.5) )
#plt.show()
fig.savefig('nav.eps')
"""
