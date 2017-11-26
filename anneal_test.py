import matplotlib.pyplot as plt
defaultMetaEpsilon = 1
defaultAnnealSteps = 50000
defaultEndEpsilon = 0.1
defaultRandomPlaySteps = 100000

def annealMetaEpsilon(stepCount):
    metaEpsilon = defaultEndEpsilon + max(0, (defaultMetaEpsilon - defaultEndEpsilon) * \
        (defaultAnnealSteps - max(0, stepCount - defaultRandomPlaySteps)) / defaultAnnealSteps)
    return metaEpsilon

res = []
for i in range(1000000):
	res.append(annealMetaEpsilon(i))

plt.plot(res)
plt.ylabel('anneal')
plt.show()