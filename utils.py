import numpy as np
def oneHot(goal):
    vec = np.zeros(3)
    vec[goal] = 1
    return vec

def reshape(state):
    return np.reshape(state, (1, -1))