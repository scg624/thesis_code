import numpy as np
import os
from multiprocessing import Pool
from scipy.stats import ttest_ind
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer
import sys
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from skfeature.function.similarity_based import SPEC

from sklearn.svm import SVC,LinearSVC
import scipy.io as sio
from joblib import Parallel, delayed
import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from openpyxl import Workbook
from Utility.PerformanceMetrics import SenSpe,Metrics
from sklearn.metrics import roc_curve, auc


def save_discriminating_index(Index,ResultantFolder):
    Index = list(Index)
    print(type(Index))
    print(Index)

    wb = Workbook()
    sheet = wb.active
    sheet.title = "Accuracy_over_95"
    for i in range(0, len(Index)):
        sheet["A%d" % (i + 1)] = Index[i]
    wb.save(ResultantFolder+'Accuracy_over_95.xlsx')
    # wb.save(r"F:\Parkinson's disease\mask\L2_template\grey_distinguishing_feature\gm_index.xlsx")
    print('Save!OK!')

def SPECSVM_kFold(SubjectsData,SubjectsLabel,ResultantFolder,kFold,NormalizeFlag,NormalzeMode,iter,retain_features):
    if not os.path.exists(ResultantFolder):
        os.makedirs(ResultantFolder)
    # the number of subjects
    SubjectQuantity = SubjectsData.shape[0]
    # permute the data and label
    index = np.arange(SubjectQuantity)
    np.random.shuffle(index)
    SubjectsData = SubjectsData[index]
    SubjectsLabel = SubjectsLabel[index]

    # generate the index of leave one out
    partitions = np.array_split(index, kFold)

    print('******SPEC+SVM %d Fold CrossValidation Beginning******'%(kFold))
    Parallel_Quantity = 2
    # start = datetime.datetime.now()
    results = Parallel(n_jobs=Parallel_Quantity, backend="threading")(
        delayed(SPECSVM_kFold_Sub_Parallel)(SubjectsData, SubjectsLabel, partitions, cv, ResultantFolder,NormalizeFlag,NormalzeMode,retain_features) for cv in
        range(0,kFold))
    # end = datetime.datetime.now()
    # runtime = end-start
    # print(runtime)
    CvIdx = np.array([item[0] for item in results])
    Accuracies = np.array([item[1] for item in results])
    Sensitivities = np.array([item[2] for item in results])
    Specifities = np.array([item[3] for item in results])
    Aucs = np.array([item[4] for item in results])
    Index = np.array([item[5] for item in results])
    # fpr = np.array([item[6] for item in results])
    # tpr = np.array([item[7] for item in results])

    ResultLOOCV = {'Accuracies':Accuracies,'Sensitivities':Sensitivities,'Specifities':Specifities,'Aucs':Aucs}
    ResultantFile = ResultantFolder + 'Accuracies_'+str(iter)+'iter.mat'
    sio.savemat(ResultantFile,ResultLOOCV)
    Resultindex = {'Index': Index}
    ResultantFile1 = ResultantFolder + 'Index'+str(iter)+'iter.mat'
    sio.savemat(ResultantFile1, Resultindex)

    # Resultindex = {'roc_auc': roc_auc}
    # ResultantFile1 = ResultantFolder + 'roc_auc'+str(iter)+'iter.mat'
    # sio.savemat(ResultantFile1, Resultindex)
    #
    # Resultindex = {'fpr': fpr}
    # ResultantFile1 = ResultantFolder + 'fpr'+str(iter)+'iter.mat'
    # sio.savemat(ResultantFile1, Resultindex)
    #
    # Resultindex = {'tpr': tpr}
    # ResultantFile1 = ResultantFolder + 'tpr'+str(iter)+'iter.mat'
    # sio.savemat(ResultantFile1, Resultindex)



    print('Average accuracy= %f \t Average Sensitivity=%f \t Average Speifity=%f\t Average Auc=%f\t'%(np.mean(Accuracies),np.mean(Sensitivities),np.mean(Specifities),np.mean(Aucs)))

    print('******SPEC+SVM %d Fold CrossValidation End******' % (kFold))

    return np.mean(Accuracies),np.mean(Sensitivities),np.mean(Specifities),np.mean(Aucs)



def SPECSVM_kFold_Sub_Parallel(SubjectsData,SubjectsLabel,partitions,cv,ResultantFolder,NormalizeFlag,NormalzeMode,retain_features):

    # partition = partitions[cv]
    # the sample index for train and test
    # TRAIN = np.delete(partitions, partition)
    # TRAIN = np.delete(partitions, cv, axis=0)
    # TRAIN = TRAIN.flatten()

    TRAIN = []
    for i in range(len(partitions)):
        if i!=cv:
            TRAIN.extend(partitions[i])

    TEST = partitions[cv]

    # divid train and test samples
    X_train = SubjectsData[TRAIN, :]
    y_train = SubjectsLabel[TRAIN]

    X_test = SubjectsData[TEST, :]
    y_test = SubjectsLabel[TEST]
    # print('X_train.shape',X_train.shape)


    # normalization
    if NormalizeFlag==True:
        if NormalzeMode == 'MinMaxScaler':
            normalize = MinMaxScaler()
        elif NormalzeMode == 'StandardScaler':
            normalize = StandardScaler()
        elif NormalzeMode == 'Normalizer':
            normalize = Normalizer()
        X_train = normalize.fit_transform(X_train)
        X_test = normalize.transform(X_test)

    clf = LinearSVC(C=1, class_weight='balanced')

    score = SPEC.spec(X_train)  # 计算每一个变量的得分
    feature_index = SPEC.feature_ranking(score)  # 依据变量得分选择变量
    # print(feature_index)

    # obtain the dataset on the selected features
    X_train_new = X_train[:, feature_index[0:retain_features]]
    X_test_new = X_test[:, feature_index[0:retain_features]]
    print("after X_train reduce_dim:",X_train_new.shape)
    print("after X_test reduce_dim:",X_test_new.shape)

    Index = feature_index[0:retain_features]
    # print(Index)

    # train a classification model with the selected features on the training dataset
    clf.fit(X_train_new, y_train)

    # predict the class labels of test data
    PredictLabel = clf.predict(X_test_new)

    # obtain the classification accuracy on the test data
    Accuracy = accuracy_score(y_test,PredictLabel)
    Auc = roc_auc_score(y_test,PredictLabel)

    # fpr, tpr, thresholds = roc_curve(y_test, PredictLabel)
    # roc_auc = auc(fpr, tpr)

    Sensitivity, Specifity = SenSpe(y_test, PredictLabel)
    print('The %d iteration!\t Accuracy=%f \t Sensitivity=%f \t Specifity=%f\t Auc=%f\t' % ((cv + 1), Accuracy,Sensitivity,Specifity,Auc))# for kFold
    # LogFile = ResultantFolder+'worklog.txt'
    # with open(LogFile,'a') as file:
    #     for i in
    # if Accuracy >= 0.95:
    #     save_discriminating_index(Index,ResultantFolder)

    return cv+1,Accuracy,Sensitivity,Specifity,Auc,Index


