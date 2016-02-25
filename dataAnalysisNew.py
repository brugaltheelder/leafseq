import matplotlib

matplotlib.use('Agg')

import numpy as np
import numpy.matlib as matlib
import scipy.io as io
import matplotlib.pyplot as plt
import pandas as pd
import itertools


def applyPlotStyle():
    plt.xlabel('Objective Function Ratio to CLO_DLO-U')
    plt.ylabel('Ratio of Total MUs to CLO_DLO-U')
    plt.xlim(0.1, 1000)
    plt.ylim(0, 10)
    plt.xscale('log')


# constants
marker = itertools.cycle(('8', '+', '.', 'o', '*', 'v', '^', 'p', 's', 'D'))

# run parameters
makePlots = True
makePlotsObjVsMUs = True
outputExcel = True
excelOutFilename = 'outputNew.xlsx'
savefolder = '/media/troy/DataDrive/Dropbox/PythonLS/leafseq/figsClinKnew/'
testRatioPosition = 13
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

headings = []

# for each type, load in matrix, append to values
tups = []
for i in range(len(baseFolders)):
    F = io.loadmat(baseFolders[i] + matNames[i])
    inner = np.append(np.round(F['paramList'].transpose(), decimals=2),
                      matlib.repmat(indexTypes[i], 1, len(F['paramList'])), axis=0).tolist()
    tups = tups + list(zip(*inner))

    if i == 0:
        headings = F['runTags'][0]

        objs = F['objectives']
        mus = F['totalMU']
    else:
        objs = np.append(objs, F['objectives'], axis=0)
        mus = np.append(mus, F['totalMU'], axis=0)
print len(tups)
print objs.shape
print tups

seedHeadings = [h for h in headings if 'CLO_' not in h]
seedIndices = [i for i in range(len(headings)) if 'CLO_' not in headings[i]]
cloHeadings = [h for h in headings if 'CLO_' in h]
cloIndices = [i for i in range(len(headings)) if 'CLO_' in headings[i]]

# build dataframe
rowtitles = ['K', 'Param1', 'Param2', 'TargetF']
index = pd.MultiIndex.from_tuples(tups, names=rowtitles)
objs_df = pd.DataFrame(objs, index=index, columns=headings)

# calc objs scaled and mus scaled
objs_scaled = np.copy(objs)
mus_scaled = np.copy(mus)
objs_scaled_row = np.copy(objs_scaled[:, testRatioPosition])
mus_scaled_row = np.copy(mus_scaled[:, testRatioPosition])
for i in range(objs_scaled.shape[1]):
    objs_scaled[:, i] = objs_scaled[:, i] / objs_scaled_row
    mus_scaled[:, i] = mus_scaled[:, i] / mus_scaled_row
objs_scaled_df = pd.DataFrame(objs_scaled, index=index, columns=headings)
mus_scaled_df = pd.DataFrame(mus_scaled, index=index, columns=headings)

print objs_scaled_df.describe().apply(pd.Series.round, args=(3,))
print mus_scaled_df.describe().apply(pd.Series.round, args=(3,))

# plot objs_scaledSEEDS
if makePlots:
    objs_scaled_df.plot(x=np.arange(objs_scaled.shape[0]), y=seedHeadings, kind='box',
                        title="Objective function ratios to CLO_DLO-U", meanline=True, showmeans=True, logy=True)
    plt.plot((0, 2 + len(headings)), (1, 1), color='c', zorder=1)
    plt.xlabel('Method')
    plt.ylabel('Ratio to CLO_DLO-U Objective')
    fig = plt.gcf()
    fig.set_size_inches(12, 8, forward=True)
    plt.savefig(savefolder + 'FullObjRatioBoxPlotSEED.png', bbox_inches='tight', dpi=400)
    plt.show()

# plot objs_scaled CLO
if makePlots:
    objs_scaled_df.plot(x=np.arange(objs_scaled.shape[0]), y=cloHeadings, kind='box',
                        title="Objective function ratios to CLO_DLO-U", meanline=True, showmeans=True, logy=True)
    plt.plot((0, 2 + len(headings)), (1, 1), color='c', zorder=1)
    plt.xlabel('Method')
    plt.ylabel('Ratio to CLO_DLO-U Objective')
    fig = plt.gcf()
    fig.set_size_inches(12, 8, forward=True)
    plt.savefig(savefolder + 'FullObjRatioBoxPlotCLO.png', bbox_inches='tight', dpi=400)
    plt.show()

# output latex table of stats
with open(savefolder + 'objs_scaled_df_stats.txt', 'w') as f:
    f.write(objs_scaled_df.describe().apply(pd.Series.round, args=(2,)).to_latex())

with open(savefolder + 'objs_scaled_df_stats_cleaned.txt', 'w') as f:
    f.write(objs_scaled_df[objs_scaled_df[:] < cutoff].describe().apply(pd.Series.round, args=(2,)).to_latex())

# plot MUs scaled
if makePlots:
    mus_scaled_df.plot(x=np.arange(mus_scaled.shape[0]), y=cloHeadings, kind='box', title="MU ratios to CLO_DLO-U",
                       meanline=True, showmeans=True)
    plt.plot((0, 2 + len(headings)), (1, 1), color='c', zorder=1)
    plt.xlabel('Method')
    plt.ylabel('MU Ratio to CLO_DLO-U Total MUs')
    fig = plt.gcf()
    fig.set_size_inches(12, 8, forward=True)
    plt.savefig(savefolder + 'FullMURatiosCLO.png', bbox_inches='tight', dpi=400)
    plt.show()

# output latex table of stats
with open(savefolder + 'mus_scaled_df_stats.txt', 'w') as f:
    f.write(mus_scaled_df.describe().apply(pd.Series.round, args=(2,)).to_latex())

with open(savefolder + 'mus_scaled_df_stats_cleaned.txt', 'w') as f:
    f.write(mus_scaled_df[mus_scaled_df[:] < cutoff].describe().apply(pd.Series.round, args=(2,)).to_latex())

# make objective function comparisons
if makePlots:
    # build percent improvement for DLO, DLO-U, Conv
    obj_imp = np.divide(F['objectives'][:, seedIndices], F['objectives'][:, cloIndices])
    obj_improvement_df = pd.DataFrame(obj_imp, columns=cloHeadings)
    print obj_improvement_df.median()
    print objs_scaled_df[cloHeadings].median()
    objImpVsScaled = pd.concat([obj_improvement_df.median(), objs_scaled_df[cloHeadings].median()], axis=1)
    with open(savefolder + 'improvement table.txt', 'w') as f:
        f.write(objImpVsScaled.apply(pd.Series.round, args=(2,)).transpose().to_latex())

    pass

# make the obj vs mus plot
if makePlotsObjVsMUs:
    fig = plt.figure()
    fig.set_size_inches(10, 8, forward=True)
    ax = plt.gca()
    p1 = ['CLO_DLO-U', 'DLO      ', 'Conv     ', 'DLO-U    ']
    # TODO FILL IN THE COMPARISONS
    # p1 = ['DLO-U','a.','b.','c.','d.']
    p2 = ['CLO_DLO-U', 'CLO_DLO  ', 'CLO_Conv ']
    p3 = ['CLO_DLO-U', 'CLO_Rand ', 'CLO_Unif ']
    p4 = ['CLO_DLO-U', 'CLO_Cent ', 'CLO_Peak ']
    pblocks = [p1, p2, p3, p4]
    positions = [221, 222, 223, 224];
    for p in range(len(pblocks)):
        plt.subplot(positions[p])
        applyPlotStyle()
        colors = itertools.cycle(('r', 'c', 'darkblue', 'maroon', 'black', 'gray', 'g', 'peru'))
        marker = itertools.cycle(('D', '8', '+', '.', 'o', '*', 'v', '^', 'p', 's'))
        for c in pblocks[p]:
            plt.scatter(objs_scaled_df[c], mus_scaled_df[c], marker=marker.next(), color=colors.next(), alpha=0.35,
                        label=c, zorder=1 if c == 'CLO_DLO-U' else 0)
        plt.legend(markerscale=0.9, fontsize=12, fancybox=True, framealpha=0.7, loc='upper left')

    plt.tight_layout()
    plt.savefig(savefolder + 'ObjVsMus.png', bbox_inches='tight', dpi=400)
    plt.show()

if outputExcel:
    writer = pd.ExcelWriter(excelOutFilename)
    objs_scaled_df.to_excel(writer, 'Sheet1')
    mus_scaled_df.to_excel(writer, 'Sheet2')
    writer.save()
