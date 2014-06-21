from pylab import *
import numpy as np

labels = ['EU', 'AS', 'US']
fracs = [50, 5, 45]

def piechart(lables, values):
	figure(1, figsize=(6,6))
	explode=[0.1]*len(values)
	pie(values, explode=explode, labels=lables,
                autopct='%1.1f%%', shadow=True, startangle=90, colors=('0.85','0.75','0.65','0.55'))
	savefig('latex\pie.eps')

def sector_chart(lables, values):
	y_pos = np.arange(len(labels))
	barh(values, align='center', edgecolor='none')
	yticks(y_pos,lables)
	show()

sector_chart(labels,fracs)