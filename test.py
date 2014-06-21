import matplotlib.pyplot as plt
from pandas import *

plt.figure()

df2 = DataFrame(rand(10, 4), columns=['a', 'b', 'c', 'd'])
df2.plot(kind='bar');