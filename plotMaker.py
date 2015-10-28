import matplotlib
matplotlib.use('Agg')

import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import pandas as pd



def getAllPlots(savefolder, basefolder, matname, targetType, headings, l1step):
	# open data file
	F = io.loadmat(basefolder + matname)
	# build multi-index
	rowtitles = F['paramNames']
	inner = np.round(F['paramList'].transpose(),decimals=2).tolist()
	tups = list(zip(*inner))
	index = pd.MultiIndex.from_tuples(tups,names=rowtitles)	
	
	# build obj dataframe
	objs_scaled = F['objectives']
	scaledrow = np.copy(objs_scaled[:,7])
	for i in range(objs_scaled.shape[1]):
			objs_scaled[:,i] = objs_scaled[:,i]/scaledrow
	objs_scaled_df = pd.DataFrame(objs_scaled, index = index, columns=headings)

	# plot obj ratios
	objs_scaled_df.plot(x=np.arange(objs_scaled.shape[0]), kind='box', title="Objective function ratios to CG Unit for f="+targetType,meanline=True, showmeans=True, logy=True)
	plt.plot((0,8),(1,1),color='c')
	plt.xlabel('Method')
	plt.ylabel('Ratio to CG Unit objective')
	fig = plt.gcf()
	fig.set_size_inches(12,8, forward = True)
	plt.savefig(savefolder + targetType.replace(' ','_') + '_' + 'ObjRatioBoxPlot.png', bbox_inches='tight', dpi=400)
	plt.show()

	# build runtimes dataframe
	runTime_df = pd.DataFrame(F['runTimes'], index = index, columns=headings)

	# plot runtimes
	runTime_df.plot(x=np.arange(objs_scaled.shape[0]), kind='box', title="Run time in seconds per method for f="+targetType, meanline=True,showmeans=True)
	plt.xlabel('Method')
	plt.ylabel('Run time in seconds')
	fig = plt.gcf()
	fig.set_size_inches(12,8, forward = True)
	plt.savefig(savefolder + targetType.replace(' ','_') + '_' + 'RunTimesPlot.png', bbox_inches='tight', dpi=400)
	plt.show()

	# build MUs dataframe
	mus_scaled = F['totalMU']
	scaledMUrow = np.copy(mus_scaled[:,7])
	for i in range(mus_scaled.shape[1]):
			mus_scaled[:,i] = mus_scaled[:,i]/scaledMUrow
	mus_scaled_df = pd.DataFrame(mus_scaled, index = index, columns=headings)

	# plot MU ratios
	mus_scaled_df.plot(x=np.arange(objs_scaled.shape[0]), kind='box', title="MU ratios to CG Unit for f="+targetType,meanline=True, showmeans=True)
	plt.xlabel('Method')
	plt.ylabel('MU Ratio to CG Unit')
	fig = plt.gcf()
	fig.set_size_inches(12,8, forward = True)
	plt.savefig(savefolder + targetType.replace(' ','_') + '_' + 'MURatios.png', bbox_inches='tight', dpi=400)
	plt.show()

	# set plot attributes for obj vs mu
	kFix = objs_scaled_df.index.levels[0][0::2]
	l1Fix = objs_scaled_df.index.levels[1][0::l1step]
	
	# make big plot
	counter = 1
	for k in range(len(kFix)):
		for l in range(len(l1Fix)):
			ax = plt.subplot(len(kFix),len(l1Fix),counter)
			ax.set_yscale('log')
			rainbow = ['r','c','darkblue','maroon', 'yellow', 'black','gray','g', 'peru']
			for s,i in zip(objs_scaled_df.columns.values,range(len(objs_scaled_df.columns.values))):
				color = rainbow[i]
				ax.plot(mus_scaled_df.xs((kFix[k],l1Fix[l]),level=(0,1))[s],objs_scaled_df.xs((kFix[k],l1Fix[l]),level=(0,1))[s],color = color, label = s,marker = 'o', zorder =2)
			if counter==1:                        
				#ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)        
				ax.legend(markerscale = 0.5,fontsize=12,fancybox=True,framealpha=0.5)
			plt.xlabel('MU Ratio')
			plt.ylabel('Obj Ratio')
			plt.title('K = ' + str(kFix[k]) +', ' + rowtitles[1].strip() + ' = ' + str(l1Fix[l]))
			plt.plot((0,mus_scaled_df.xs((kFix[k],l1Fix[l]),level=(0,1)).max().max()),(1,1),color='orange', zorder=1)
			plt.plot((1,1),(objs_scaled_df.xs((kFix[k],l1Fix[l]),level=(0,1)).min().min(),objs_scaled_df.xs((kFix[k],l1Fix[l]),level=(0,1)).max().max()),color= 'orange', zorder=1)
			plt.axis('tight')
			plt.xlim(min(0.5,mus_scaled_df.xs((kFix[k],l1Fix[l]),level=(0,1)).min().min()),mus_scaled_df.xs((kFix[k],l1Fix[l]),level=(0,1)).max().max())
			plt.ylim(min(0.7,objs_scaled_df.xs((kFix[k],l1Fix[l]),level=(0,1)).min().min()),objs_scaled_df.xs((kFix[k],l1Fix[l]),level=(0,1)).max().max())
			counter+=1
	fig = plt.gcf()
	fig.set_size_inches(5 * len(l1Fix),3 * len(kFix), forward = True)
	plt.tight_layout(pad = 0.3)
	plt.savefig(savefolder + targetType.replace(' ','_') + '_' + 'ObjVsMU.png', bbox_inches='tight', dpi=400)
	plt.show()





savefolder = '/media/troy/DataDrive/Dropbox/PythonLS/leafseq/figs/'
basefolders = ['stepOrder2/','stepOrder0/','doublesin/','singlesin/','erfs/']
matnames = ['stepFunctionOutOrder2.mat','stepFunctionOutOrder0.mat','doublesinout.mat','sinout.mat','erfsout.mat']
targetTypes = ['Spline Step','Unit Step','Double Sine','Single Sine','Error Functions']
headingList = ['Random','SW','Centered', 'Peaks','CG Erf Seed', 'CG Erf','CG Unit Seed', 'CG Unit']
l1FixStep = [2,2,2,1,2]

for i in range(len(basefolders)):
	getAllPlots(savefolder, basefolders[i], matnames[i], targetTypes[i], headingList, l1FixStep[i])
# i=3
# getAllPlots(savefolder, basefolders[i], matnames[i], targetTypes[i], headingList, l1FixStep[i])