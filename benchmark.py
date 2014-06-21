__author__ = 'thoru_000'

from datetime import *
import ystockquote


def getBench(f_tdelta,bench):

	sDate = datetime.today().date() + timedelta(days=-f_tdelta)
	sDate = sDate.strftime("%Y-%m-%d")
	eDate = datetime.today().date()
	eDate = eDate.strftime("%Y-%m-%d")

	tdate = []
	temp = []

	print "Getting prices for " + str(bench)
	temp.append(ystockquote.get_historical_prices(bench, sDate, eDate))

	for y in temp[0]:
		tdate.append(y)

	tdate = sorted(tdate, reverse=True)
	ptemp = [[0 for x in xrange(1)] for x in xrange(len(tdate))]

	for datee in range(len(tdate)):
		for assetnum in range(1):
			try:
				q = temp[assetnum][tdate[datee]]['Close']
				ptemp[datee][assetnum] = q
			except KeyError:
				print "wiww"
				ptemp[datee][assetnum] = ptemp[datee-1][assetnum]

	return ptemp