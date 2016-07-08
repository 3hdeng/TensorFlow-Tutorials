from functions import test

n_features = 2
n_clusters = 3


test(n_features, n_clusters)


"""
strange importError when using matplotlib.pyplot in ipynb

NameError: global name 'plt' is not defined

ImportError: cannot import name plot_cluster
-->  You have circular dependent imports


--> add an extra cell in the beginning of the notebook
%matplotlib inline
import matplotlib.pyplot as plt
plt.plot([1,2,3,4])

"""


