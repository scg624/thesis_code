from Utility.PrepareData import load2DData,loadMask
from ClassifyFunc.LASSOSTAB_SVM_CV import LASSOSTABSVM_kFold
from ClassifyFunc.SVM_CV  import SVM_kFold
import numpy as np
import scipy.io as sio
import warnings
warnings.filterwarnings("ignore")


# load mask
GreyMaskDir = '..\PD_HC_Smooth/age_50_80/two_sample_ttest\GM_uncorrected_p05_100\gm_combine_spmT_0001_0002_size100_p05.img'
# GreyMaskDir = '..\PD_HC_Smooth/age_50_80/two_sample_ttest\WM_uncorrected_p05_100\wm_combine_spmT_0001_0002_size100_p05.img'

GreyMask = loadMask(GreyMaskDir)

# load structual data with 3D
DataDir = '..\PD_HC_Smooth/age_50_80/'
# GroupName = ['hc_gm50_80', 'pd_gm50_80']
GroupName = ['hc_wm50_80', 'pd_wm50_80']

Group1Dir = DataDir + GroupName[0]
Group2Dir = DataDir + GroupName[1]


Group1Data = load2DData(Group1Dir, GreyMask)
Group2Data = load2DData(Group2Dir, GreyMask)
# print(Group1Data.shape[0])
# # for test
# Group1Data = Group1Data[0:Group2Data.shape[0],:]
# # for testx

SubjectsData = np.concatenate((Group1Data, Group2Data), axis=0)

Group1Label = -1 * np.ones([Group1Data.shape[0]])

# # for test,
# Group1Label = -1 * np.ones([Group2Data.shape[0]])
# # for test

Group2Label = np.ones([Group2Data.shape[0]])
SubjectsLabel = np.concatenate((Group1Label,Group2Label), axis=0)
# **********************************************************************************************************************
# **********************************************************************************************************************
# stability_selection + svm
ResultantFolder = 'Results/WM_Stability_LASSO/'
kFold = 5
nIter = 10
NormalizeFlag = 'True'# True/False
NormalzeMode = 'MinMaxScaler'#StandardScaler/MinMaxScaler/Normalizer/None

Accuracies = np.zeros([nIter,1])
Sensitivities = np.zeros([nIter,1])
Specificities = np.zeros([nIter,1])
for i in range(nIter):
     acc,sen,spe= LASSOSTABSVM_kFold(SubjectsData,SubjectsLabel,ResultantFolder,kFold,NormalizeFlag,NormalzeMode,i)
     Accuracies[i, 0] = acc
     Sensitivities[i, 0] = sen
     Specificities[i, 0] = spe

print('After %d iterations, the mean accuracies = %f \t the mean sensitivities = %f \t the mean specificities = %f \t'%(nIter,np.mean(Accuracies),np.mean(Sensitivities),np.mean(Specificities)))
ResultKfold = {'Accuracies':np.mean(Accuracies),'Sensitivities':np.mean(Sensitivities),'Specifities':np.mean(Specificities)}
ResultantFile = ResultantFolder + 'Final_performance.mat'
sio.savemat(ResultantFile,ResultKfold)
# **********************************************************************************************************************
# # **********************************************************************************************************************
