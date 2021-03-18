import numpy as np
import os
from multiprocessing import Pool
from scipy.stats import ttest_ind
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer
import sys
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn import linear_model
from sklearn.svm import SVC,LinearSVC,SVR
import scipy.io as sio
from joblib import Parallel, delayed
import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from skrebate import ReliefF
from skrebate import SURF
from skrebate import SURFstar
from skrebate import MultiSURF
from skrebate import MultiSURFstar
from skrebate import TuRF
from sklearn.feature_selection import RFE
from  sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron

from sklearn.linear_model import LogisticRegression


from Utility.PerformanceMetrics import SenSpe,Metrics
from sklearn import tree
from openpyxl import Workbook




def RFESVM_kFold(SubjectsData,SubjectsLabel,ResultantFolder,kFold,NormalizeFlag,NormalzeMode,iter,retain_features):
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

    print('******RFE+SVM %d Fold CrossValidation Beginning******'%(kFold))
    Parallel_Quantity = 2
    # start = datetime.datetime.now()
    results = Parallel(n_jobs=Parallel_Quantity, backend="threading")(
        delayed(RFESVM_kFold_Sub_Parallel)(SubjectsData, SubjectsLabel, partitions, cv, ResultantFolder,NormalizeFlag,NormalzeMode,retain_features) for cv in
        range(0,kFold))

    # end = datetime.datetime.now()
    # runtime = end-start
    # print(runtime)
    CvIdx = np.array([item[0] for item in results])
    Accuracies = np.array([item[1] for item in results])
    Sensitivities = np.array([item[2] for item in results])
    Specifities = np.array([item[3] for item in results])
    Auc = np.array([item[4] for item in results])
    Index = np.array([item[5] for item in results])

    ResultLOOCV = {'Accuracies':Accuracies,'Sensitivities':Sensitivities,'Specifities':Specifities,'AUC':Auc}
    ResultantFile = ResultantFolder + 'Accuracies_'+str(iter)+'iter.mat'
    sio.savemat(ResultantFile,ResultLOOCV)
    Resultindex = {'Index': Index}
    ResultantFile1 = ResultantFolder + 'Index'+str(iter)+'iter.mat'
    sio.savemat(ResultantFile1, Resultindex)

    print('Average accuracy= %f \t Average Sensitivity=%f \t Average Speifity=%f\t'%(np.mean(Accuracies),np.mean(Sensitivities),np.mean(Specifities)))

    print('******RFE+SVM %d Fold CrossValidation End******' % (kFold))

    return np.mean(Accuracies),np.mean(Sensitivities),np.mean(Specifities),np.mean(Auc)



def RFESVM_kFold_Sub_Parallel(SubjectsData,SubjectsLabel,partitions,cv,ResultantFolder,NormalizeFlag,NormalzeMode,retain_features):
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


    estimator = LinearSVC(C=1, class_weight='balanced')

    classifier = LinearSVC(C=1, class_weight='balanced')

    # estimator = LinearSVC(C=0.1, class_weight='balanced')
    model = RFE(estimator, n_features_to_select=retain_features, step=0.1).fit(X_train,y_train)
    X_train_new = model.transform(X_train)
    X_test_new = model.transform(X_test)
    # print(X_train_new.shape)
    # print("X_test_new:",X_test_new.shape)
    Index = model.get_support(indices=True)
    # print(len(Index))
    # print(type(Index))
    # print(type(Index[0]))
    # selector = make_pipeline(RFE(estimator, n_features_to_select=retain_features, step=0.01),
    #                          classifier)

    classifier.fit(X_train_new,y_train)
    PredictLabel = classifier.predict(X_test_new)
    Accuracy = accuracy_score(y_test,PredictLabel)
    Sensitivity, Specifity = SenSpe(y_test, PredictLabel)
    auc = roc_auc_score(y_test,PredictLabel)
    print('The %d iteration!\t Accuracy=%f \t Sensitivity=%f \t Specifity=%f\t AUC=%f\t' % ((cv + 1), Accuracy,Sensitivity,Specifity,auc))# for kFold


    # wb = Workbook()
    # sheet = wb.active
    # sheet.title = "gm_tpm"
    # for i in range(0, len(Index)):
    #     sheet["A%d" % (i + 1)] = Index[i]
    # ResultantFolder = os.path.join('key_brain/index' % (),'.xlsx')
    # wb.save(r"F:\Parkinson's disease\mask\L2_template\grey_distinguishing_feature\gm_index46.xlsx")
    # print('Save!OK!')

    LogFile = ResultantFolder+'worklog.txt'

    # with open(LogFile,'a') as file:
    #     for i in
    return cv+1,Accuracy,Sensitivity,Specifity,auc,Index
