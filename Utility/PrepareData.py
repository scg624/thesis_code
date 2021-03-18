import numpy as np
import os
import nibabel as nib
from scipy.stats import ttest_ind,stats,levene

# import warnings
# warnings.filterwarnings('error')

########################################################
def readFile(filepath):
    # get the file list of filepath
    files = os.listdir(filepath)
    map_filenames = []
    for file in files:
        if not os.path.isdir(file):
            fullpath = os.path.join(filepath, file)
            map_filenames.append(fullpath)
    return map_filenames


def load3DData(dataDir,sortBegin,sortEnd):
    #*******************************************************
    # input:
    # load neuroimaging data
    # dataDir : the data directory
    # locate the number in file name to sort files
    # sortBegin : the begin location of the number
    # sortEnd : the end location of the number
    # output:
    # data : the data with the shape of [M,N,K,L]
    # [M,N,K] - the 3D of structual MRI
    # L - the number of data
    # *******************************************************

    # list files and sort them
    fileList = []
    for file in os.listdir(dataDir):
        if file.endswith('.nii'):
            fileList.append(file)
    fileList.sort(key=lambda x: int(x[sortBegin:sortEnd]))

    #load 3D data
    img = nib.load(dataDir + '/' + fileList[0])
    imgData = img.get_data()
    print(np.shape(imgData))
    data = np.zeros([imgData.shape[0],imgData.shape[1],imgData.shape[2], len(fileList)])
    data[:, :, :, 0] = imgData
    for i in range(1,len(fileList)):
        img = nib.load(dataDir + '/' + fileList[i])
        imgData = img.get_data()
        data[:, :, :, i] = imgData
    return data

def load2DData(DataDir,Mask,sortBegin=None,sortEnd=None):
    #*******************************************************
    # input:
    # load neuroimaging data
    # dataDir : the data directory
    # locate the number in file name to sort files
    # sortBegin : the begin location of the number
    # sortEnd : the end location of the number
    # output:
    # data : the data with the shape of [L,M*N*K]
    # [L] - the 3D of structual MRI
    # M*N*K - the number of data
    # *******************************************************

    # list files and sort them
    FileList = []
    for file in os.listdir(DataDir):
        if file.endswith('.nii'):#Determine if the function ends in .nii.
            FileList.append(file)
    # print(FileList)
    if (sortBegin!=None and sortEnd!=None):
    #     FileList.sort(key=lambda  x:int(x[:-4]))
    # else:
        FileList.sort(key=lambda x: int(x[sortBegin:sortEnd]))#

    #load 2D data
    # print(FileList)
    # print(len(FileList))
    FeatureQuantity = np.sum(Mask!=0)
    print(FeatureQuantity)
    Data = np.zeros([len(FileList),FeatureQuantity])

    for i in range(0,len(FileList)):
        Img = nib.load(DataDir + '/' + FileList[i])
        ImgData = Img.get_data()
        Data[i,:] = np.transpose(ImgData[np.where(Mask != 0)])

    return Data

def load2DData_(DataDir,sortBegin=None,sortEnd=None):
    #*******************************************************
    # input:
    # load neuroimaging data
    # dataDir : the data directory
    # locate the number in file name to sort files
    # sortBegin : the begin location of the number
    # sortEnd : the end location of the number
    # output:
    # data : the data with the shape of [L,M*N*K]
    # [L] - the 3D of structual MRI
    # M*N*K - the number of data
    # *******************************************************

    # list files and sort them
    FileList = []
    for file in os.listdir(DataDir):
        if file.endswith('.nii'):#Determine if the function ends in .nii.
            FileList.append(file)
    # print(FileList)
    if (sortBegin!=None and sortEnd!=None):
    #     FileList.sort(key=lambda  x:int(x[:-4]))
    # else:
        FileList.sort(key=lambda x: int(x[sortBegin:sortEnd]))#

    #load 2D data
    # print(FileList)
    # print(len(FileList))


    Data = []
    for i in range(0,len(FileList)):
        Img = nib.load(DataDir + '/' + FileList[i])
        ImgData = Img.get_data()
        Data.append(((ImgData).reshape(1, -1))[0])
    Data_ = np.array(Data)
    print(Data_.shape)
    return Data_

def loadMask(maskDir):
    # *******************************************************
    # input:
    # load neuroimaging mask
    # maskDir : the mask directory
    # output:
    # the mask data
    # *******************************************************
    maskImg =nib.load(maskDir)
    return maskImg.get_data()

def rankTtest(SubjectData,SubjectLabel,Pval):
    # *******************************************************
    # input:
    # SubjectData
    # SubjectLabel
    # Pval: the threshold of p value
    # output:
    # the ttest mask
    # *******************************************************
    Group1Data = SubjectData[np.where(SubjectLabel==-1)[0],:]
    Group2Data = SubjectData[np.where(SubjectLabel==1)[0],:]

    T,P = ttest_ind(Group1Data,Group2Data,axis=0,equal_var=False,nan_policy='omit')
    P[np.isnan(P)]=0
    TtestMask = np.zeros_like(P)
    TtestMask[np.where(P<=Pval)]=1
    # print(np.sum(TtestMask))
    return TtestMask

def two_ttest(Group1Data,Group2Data,Pval):
    # *******************************************************
    # input:
    # SubjectData
    # SubjectLabel
    # Pval: the threshold of p value
    # output:
    # the ttest mask
    # *******************************************************
    stat,p_vaule = levene(Group1Data, Group2Data)
    if p_vaule > 0.05:
        T, P = ttest_ind(Group1Data, Group2Data,nan_policy='omit')
        # 如果方差不齐性， 设置equal_var = False
    else:
        T, P = ttest_ind(Group1Data, Group2Data, equal_var=False,nan_policy='omit')
        # 如果方差不齐性， 设置equal_var = False
    if P <= 0.05:
        sig = '显著'
        print(sig)
        # 如果t检验结果 p值 <= 0.05，sig赋值为 '显著'
    P[np.isnan(P)]=0
    TtestMask = np.zeros_like(P)
    TtestMask[np.where(P<=Pval)]=1
    print(np.sum(TtestMask))
    return TtestMask





