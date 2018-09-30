
scores = [3.0, 1.0, 0.2]

import numpy as np

def softmax(x):
	return np.exp(x)/np.sum(np.exp(x),0)    #"""Compute softmax values for each sets of scores in x."""
	pass  # TODO: Compute and return softmax(x)


print (softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
j