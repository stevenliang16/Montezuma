import numpy as np
def oneHot(goal):
    vec = np.zeros((1,6))
    vec[0, goal] = 1
    return vec

def reshape(state):
    return np.reshape(state, (1, -1))