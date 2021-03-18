from sklearn.metrics import f1_score,precision_score,confusion_matrix,recall_score,roc_auc_score

def SenSpe(TrueLabels, PredictLabels):
    cm = confusion_matrix(TrueLabels, PredictLabels)
    Sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    Specifity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    return Sensitivity,Specifity

def Metrics(TrueLabels, PredictLabels):
    F1 = f1_score(TrueLabels, PredictLabels)
    Precision = precision_score(TrueLabels,PredictLabels)
    Recall = recall_score(TrueLabels,PredictLabels)
    Auc = roc_auc_score(TrueLabels,PredictLabels)
    return F1,Precision,Recall,Auc
