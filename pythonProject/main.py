# Ingegneria della conoscenza

import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pgmpy

from numpy import mean
from numpy import std
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from pgmpy.estimators import K2Score, HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from sklearn import svm
from sklearn.svm import SVC

Cancer = pd.read_csv("C:\\Users\\natax\\PycharmProjects\\pythonProject\\smoking.csv")
print()
print()
print("Benvenuto nel nostro sistema per predire se, presi dei soggetti, essi sono affetti o meno dal Cancro al Seno")
print()
print()
print(Cancer)
print()

# dataset di input eliminando l'ultima colonna in quanto servirà per l'output
X = Cancer.drop("smoking", axis=1)
Y = Cancer["smoking"]

# BILANCIAMENTO DELLE CLASSI
# Proporzione dei non malati di tiroide (0) e malati di tiroide (1): [Numero di (non) malati di tiroide/Numero totale di pazienti]
print()
print('non presenza fumo:',Cancer.smoking.value_counts()[0], '(% {:.2f})'.format(Cancer.smoking.value_counts()[0] /Cancer.smoking.count() * 100))
print('presenza fumo:',Cancer.smoking.value_counts()[1], '(% {:.2f})'.format(Cancer.smoking.value_counts()[1] /Cancer.smoking.count() * 100), '\n')

# Visualizzazione del grafico
Cancer['smoking'].value_counts().plot(kind='bar').set_title('smoking')
plt.show()

# Creazione della feature X e del target y
X = Cancer.to_numpy()
y = Cancer["smoking"].to_numpy() # K-Fold Cross Validation


kf = StratifiedKFold(n_splits=5)  # La classe è in squilibrio, quindi utilizzo Stratified K-Fold ???


# Classificatori per la valutazione
knn = KNeighborsClassifier()
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
svc = SVC()

# Score delle metriche
model = {
        'KNN' : {'accuracy_list': 0.0,
                 'precision_list' : 0.0,
                 'recall_list' : 0.0,
                 'f1_list' : 0.0
        },

        'DecisionTree' : {'accuracy_list': 0.0,
                                    'precision_list' : 0.0,
                                    'recall_list' : 0.0,
                                    'f1_list' : 0.0
        },

        'RandomForest' : {'accuracy_list': 0.0,
                                    'precision_list' : 0.0,
                                    'recall_list' : 0.0,
                                    'f1_list' : 0.0
        },

        'SVM' : {'accuracy_list': 0.0,
                 'precision_list' : 0.0,
                 'recall_list' : 0.0,
                 'f1_list' : 0.0
        }
}

# K-Fold dei classificatori
for train_index, test_index in kf.split(X, y):

    training_set, testing_set = X[train_index], X[test_index]

    # Dati di train
    data_train = pd.DataFrame(training_set, columns=Cancer.columns)
    X_train = data_train.drop("smoking", axis=1)
    y_train = data_train.smoking

    # Dati di test
    data_test = pd.DataFrame(testing_set, columns=Cancer.columns)
    X_test = data_test.drop("smoking", axis=1)
    y_test = data_test.smoking

    # Fit dei classificatori
    knn.fit(X_train, y_train)
    dtc.fit(X_train, y_train)
    rfc.fit(X_train, y_train)
    svc.fit(X_train, y_train)

    y_pred_knn = knn.predict(X_test)
    y_pred_dtc = dtc.predict(X_test)
    y_pred_rfc = rfc.predict(X_test)
    y_pred_SVM = svc.predict(X_test)


# Salvo le metriche del fold nel dizionario
    model['KNN']['accuracy_list'] = (metrics.accuracy_score(y_test, y_pred_knn))
    model['KNN']['precision_list'] = (metrics.precision_score(y_test, y_pred_knn))
    model['KNN']['recall_list'] = (metrics.recall_score(y_test,y_pred_knn))
    model['KNN']['f1_list'] = (metrics.f1_score(y_test, y_pred_knn))

    model['DecisionTree']['accuracy_list'] = (metrics.accuracy_score(y_test, y_pred_dtc))
    model['DecisionTree']['precision_list'] = (metrics.precision_score(y_test, y_pred_dtc))
    model['DecisionTree']['recall_list'] = (metrics.recall_score(y_test, y_pred_dtc))
    model['DecisionTree']['f1_list'] = (metrics.f1_score(y_test, y_pred_knn))

    model['RandomForest']['accuracy_list'] = (metrics.accuracy_score(y_test, y_pred_rfc))
    model['RandomForest']['precision_list'] = (metrics.precision_score(y_test, y_pred_rfc))
    model['RandomForest']['recall_list'] = (metrics.recall_score(y_test, y_pred_rfc))
    model['RandomForest']['f1_list'] = (metrics.f1_score(y_test, y_pred_rfc))

    model['SVM']['accuracy_list'] = (metrics.accuracy_score(y_test, y_pred_SVM))
    model['SVM']['precision_list'] = (metrics.precision_score(y_test, y_pred_SVM))
    model['SVM']['recall_list'] = (metrics.recall_score(y_test, y_pred_SVM))
    model['SVM']['f1_list'] = (metrics.f1_score(y_test, y_pred_SVM))
