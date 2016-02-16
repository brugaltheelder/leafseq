import matplotlib

matplotlib.use('Agg')

import numpy as np
import numpy.matlib as matlib
import scipy.io as io
import matplotlib.pyplot as plt
import pandas as pd

makePlots = False

savefolder = '/media/troy/DataDrive/Dropbox/PythonLS/leafseq/figsClinK/'

headings = ['Trad Seed', 'Trad', 'a.','b.','c.', 'd.','DLO Seed', 'DLO','DLO-U Seed', 'DLO-U']
testRatioPosition = 9
cutoff = 1e3

# read in data
baseFolders = []
matNames = []
targetTypes = []
indexTypes = []

baseFolders.append('stepOrder2ClinK/')
matNames.append('stepFunctionOutOrder2.mat')
targetTypes.append('Spline Step')
indexTypes.append('stepOrder2')

baseFolders.append('stepOrder0ClinK/')
matNames.append('stepFunctionOutOrder0.mat')
targetTypes.append('Unit Step')
indexTypes.append('stepOrder0')

baseFolders.append('doublesinClinK/')
matNames.append('doublesinout.mat')
targetTypes.append('Double Sine')
indexTypes.append('doubleSin')

baseFolders.append('singlesinClinK/')
matNames.append('sinout.mat')
targetTypes.append('Single Sine')
indexTypes.append('singleSin')

baseFolders.append('erfsClinK/')
matNames.append('erfsout.mat')
targetTypes.append('Error Functions')
indexTypes.append('erf')


# for each type, load in matrix, append to values
tups = []
for i in range(len(baseFolders)):
	F = io.loadmat(baseFolders[i] + matNames[i])
	inner = np.append(np.round(F['paramList'].transpose(),decimals=2), matlib.repmat(indexTypes[i], 1, len(F['paramList'])), axis=0).tolist()
	tups = tups + list(zip(*inner))
	
	if i==0:
		objs = F['objectives']
		mus = F['totalMU']
	else:
		objs = np.append(objs, F['objectives'],axis = 0)
		mus = np.append(mus, F['totalMU'],axis = 0)
print len(tups)
print objs.shape
print tups


# build dataframe
rowtitles = ['K','Param1','Param2','TargetF']
index = pd.MultiIndex.from_tuples(tups, names=rowtitles)
objs_df = pd.DataFrame(objs,index=index, columns = headings)


#calc objs scaled and mus scaled
objs_scaled = np.copy(objs)
mus_scaled = np.copy(mus)
objs_scaled_row = np.copy(objs_scaled[:,testRatioPosition])
mus_scaled_row = np.copy(mus_scaled[:,testRatioPosition])
for i in range(objs_scaled.shape[1]):
	objs_scaled[:,i] = objs_scaled[:,i]/objs_scaled_row
	mus_scaled[:,i] = mus_scaled[:,i]/mus_scaled_row
objs_scaled_df = pd.DataFrame(objs_scaled,index = index, columns = headings)
mus_scaled_df = pd.DataFrame(mus_scaled,index = index, columns = headings)

print objs_scaled_df.describe().apply(pd.Series.round,args=(3,))
print mus_scaled_df.describe().apply(pd.Series.round,args=(3,))



# plot objs_scaled
if makePlots:
	objs_scaled_df.plot(x = np.arange(objs_scaled.shape[0]), kind = 'box', title="Objective function ratios to DLO-U",meanline=True, showmeans=True, logy=True)
	plt.plot((0,2+len(headings)),(1,1),color='c',zorder=1)
	plt.xlabel('Method')
	plt.ylabel('Ratio to DLO-U Objective')
	fig = plt.gcf()
	fig.set_size_inches(12,8, forward = True)
	plt.savefig(savefolder + 'FullObjRatioBoxPlot.png', bbox_inches='tight', dpi=400)
	plt.show()



# output latex table of stats
with open(savefolder+'objs_scaled_df_stats.txt','w') as f:
	f.write(objs_scaled_df.describe().apply(pd.Series.round,args=(2,)).to_latex())


with open(savefolder+'objs_scaled_df_stats_cleaned.txt','w') as f:
	f.write(objs_scaled_df[objs_scaled_df[:]<cutoff].describe().apply(pd.Series.round,args=(2,)).to_latex())



# plot MUs scaled
if makePlots:
	mus_scaled_df.plot(x=np.arange(mus_scaled.shape[0]), kind='box', title="MU ratios to DLO-U",meanline=True, showmeans=True)
	plt.plot((0,2+len(headings)),(1,1),color='c', zorder=1)
	plt.xlabel('Method')
	plt.ylabel('MU Ratio to DLO-U Total MUs')
	fig = plt.gcf()
	fig.set_size_inches(12,8, forward = True)
	plt.savefig(savefolder + 'FullMURatios.png', bbox_inches='tight', dpi=400)
	plt.show()


# output latex table of stats
with open(savefolder+'mus_scaled_df_stats.txt','w') as f:
	f.write(mus_scaled_df.describe().apply(pd.Series.round,args=(2,)).to_latex())

with open(savefolder+'mus_scaled_df_stats_cleaned.txt','w') as f:
	f.write(mus_scaled_df[mus_scaled_df[:]<cutoff].describe().apply(pd.Series.round,args=(2,)).to_latex())



