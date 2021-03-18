import numpy as np
import os
from multiprocessing import Pool
from scipy.stats import ttest_ind
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer
import sys
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC,LinearSVC
import scipy.io as sio
from joblib import Parallel, delayed
import datetime
from sklearn.metrics import accuracy_score
from stability_selection import StabilitySelection, RandomizedLasso
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression

from openpyxl import Workbook
from Utility.PerformanceMetrics import SenSpe,Metrics
from sklearn import tree


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

def LASSOSTABSVM_kFold(SubjectsData,SubjectsLabel,ResultantFolder,kFold,NormalizeFlag,NormalzeMode,iter):
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

    print('******Stability+SVM %d Fold CrossValidation Beginning******'%(kFold))
    Parallel_Quantity = 2
    # start = datetime.datetime.now()
    results = Parallel(n_jobs=Parallel_Quantity, backend="threading")(
        delayed(LASSOSTABSVM_kFold_Sub_Parallel)(SubjectsData, SubjectsLabel, partitions, cv, ResultantFolder,NormalizeFlag,NormalzeMode) for cv in
        range(0,kFold))
    # end = datetime.datetime.now()
    # runtime = end-start
    # print(runtime)
    CvIdx = np.array([item[0] for item in results])
    Accuracies = np.array([item[1] for item in results])
    Sensitivities = np.array([item[2] for item in results])
    Specifities = np.array([item[3] for item in results])
    Index = np.array([item[4] for item in results])

    ResultLOOCV = {'Accuracies':Accuracies,'Sensitivities':Sensitivities,'Specifities':Specifities}
    ResultantFile = ResultantFolder + 'Accuracies_'+str(iter)+'iter.mat'
    sio.savemat(ResultantFile,ResultLOOCV)
    Resultindex = {'Index': Index}
    ResultantFile1 = ResultantFolder + 'Index'+str(iter)+'iter.mat'
    sio.savemat(ResultantFile1, Resultindex)

    print('Average accuracy= %f \t Average Sensitivity=%f \t Average Speifity=%f\t'%(np.mean(Accuracies),np.mean(Sensitivities),np.mean(Specifities)))

    print('******Stability+SVM %d Fold CrossValidation End******' % (kFold))

    return np.mean(Accuracies),np.mean(Sensitivities),np.mean(Specifities)



def LASSOSTABSVM_kFold_Sub_Parallel(SubjectsData,SubjectsLabel,partitions,cv,ResultantFolder,NormalizeFlag,NormalzeMode):

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

    # estimator = LogisticRegression(penalty='elasticnet', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
    #                                class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto',
    #                                verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
    # estimator = ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False,
    #                        max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False,
    #                        random_state=None, selection='cyclic')
    estimator = Lasso(alpha=1.0, max_iter=1000, tol=0.0001, selection='cyclic')

    Stability = StabilitySelection(base_estimator=estimator,lambda_name='alpha',threshold=0.085).fit(X_train,y_train)
    Index = Stability.get_support(indices=True)
    # print(Index)
    print(Index.shape)
    # fea = Stability.transform(X_train)
    # print(fea.shape)
    X_train_new = Stability.transform(X_train)
    X_test_new = Stability.transform(X_test)
    print("after X_train reduce_dim:",X_train_new.shape)
    print("after X_test reduce_dim:",X_test_new.shape)

    classifier = LinearSVC(C=1, class_weight='balanced')
    # pipeline = Pipeline(
    #     [('clf', classifier),
    #      ('reduce_dim', Stability)
    #      ])

    classifier.fit(X_train,y_train)
    PredictLabel = classifier.predict(X_test)
    Accuracy = accuracy_score(y_test,PredictLabel)
    Sensitivity, Specifity = SenSpe(y_test, PredictLabel)
    print('The %d iteration!\t Accuracy=%f \t Sensitivity=%f \t Specifity=%f\t ' % ((cv + 1), Accuracy,Sensitivity,Specifity))# for kFold
    # LogFile = ResultantFolder+'worklog.txt'
    # with open(LogFile,'a') as file:
    #     for i in
    # if Accuracy >= 0.95:
    #     save_discriminating_index(Index,ResultantFolder)

    return cv+1,Accuracy,Sensitivity,Specifity,Index


