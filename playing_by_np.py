import numpy as np

bandits = np.array([[0.2,0,-0.0,-5],[0.1,-5,1,0.25],[-5,5,5,5]])
print "Number of Bandit: {}".format(bandits.shape[0])
print "Number of Action: {}".format(bandits.shape[1])
print "Length: {}".format(len(bandits))
