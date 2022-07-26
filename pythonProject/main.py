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

df_smoke = pd.read_csv("C:\\Users\\natax\\icon_project\\pythonProject\\smoking.csv")
print()
print()
print("Benvenuto nel nostro sistema per predire se, presi dei soggetti, essi sono affetti o meno dal Cancro al Seno")
print()
print()
# print(df_smoke)
print()

# conversione da stringa a intero + pulizia dataframe - POTREBBE NON SERVIRE!
# df_smoke["gender"] = df_smoke["gender"].replace("M", 0)
# df_smoke["gender"] = df_smoke["gender"].replace("F", 1)

df_smoke = df_smoke.drop(
    ["ID", "gender", "eyesight(left)", "eyesight(right)", "hearing(left)", "hearing(right)", "AST", "ALT", "Gtp",
     "oral", "tartar"], axis=1)
# df_smoke["gender"] = df_smoke ["gender"].astype(int)

print(df_smoke)

# dataset di input eliminando l'ultima colonna in quanto servirà per l'output
X = df_smoke.drop("smoking", axis=1)
Y = df_smoke["smoking"]

# BILANCIAMENTO DELLE CLASSI
# Proporzione dei non malati di tiroide (0) e malati di tiroide (1): [Numero di (non) malati di tiroide/Numero totale di pazienti]
print()
print('non presenza fumo:', df_smoke.smoking.value_counts()[0],
      '(% {:.2f})'.format(df_smoke.smoking.value_counts()[0] / df_smoke.smoking.count() * 100))
print('presenza fumo:', df_smoke.smoking.value_counts()[1],
      '(% {:.2f})'.format(df_smoke.smoking.value_counts()[1] / df_smoke.smoking.count() * 100), '\n')


# Visualizzazione del grafico
def autopct(pct):  # Mostra solo i valori delle laber che sono superiori al 10%
    return ('%.2f' % pct + "%") if pct > 10 else ''


labels = ["Fumatori", "Non fumatori"]
ax = df_smoke['smoking'].value_counts().plot(kind='pie', figsize=(4, 4), autopct=autopct, labels=None)
ax.axes.get_yaxis().set_visible(False)
plt.legend(labels=labels, loc="best")
plt.show()

# EVALUATION SELECTION: K-FOLD CROSS VALIDATION

# Creazione della feature X e del target y
X = df_smoke.to_numpy()
y = df_smoke["smoking"].to_numpy()  # K-Fold Cross Validation

kf = StratifiedKFold(n_splits=5)  # La classe è in squilibrio, quindi utilizzo Stratified K-Fold ???

# Classificatori per la valutazione
knn = KNeighborsClassifier()
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
svc = SVC()

# Score delle metriche
model = {
    'KNN': {'accuracy_list': 0.0,
            'precision_list': 0.0,
            'recall_list': 0.0,
            'f1_list': 0.0
            },

    'DecisionTree': {'accuracy_list': 0.0,
                     'precision_list': 0.0,
                     'recall_list': 0.0,
                     'f1_list': 0.0
                     },

    'RandomForest': {'accuracy_list': 0.0,
                     'precision_list': 0.0,
                     'recall_list': 0.0,
                     'f1_list': 0.0
                     },

    'SVM': {'accuracy_list': 0.0,
            'precision_list': 0.0,
            'recall_list': 0.0,
            'f1_list': 0.0
            }
}

# K-Fold dei classificatori
for train_index, test_index in kf.split(X, y):
    training_set, testing_set = X[train_index], X[test_index]

    # Dati di train
    data_train = pd.DataFrame(training_set, columns=df_smoke.columns)
    X_train = data_train.drop("smoking", axis=1)
    y_train = data_train.smoking

    # Dati di test
    data_test = pd.DataFrame(testing_set, columns=df_smoke.columns)
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
    model['KNN']['recall_list'] = (metrics.recall_score(y_test, y_pred_knn))
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


# Modello di rapporto
def model_report(model):
    df_smoke_models = []

    for clf in model:
        df_smoke_model = pd.DataFrame({'model': [clf],
                                       'accuracy': [np.mean(model[clf]['accuracy_list'])],
                                       'precision': [np.mean(model[clf]['precision_list'])],
                                       'recall': [np.mean(model[clf]['recall_list'])],
                                       'f1score': [np.mean(model[clf]['f1_list'])]
                                       })

        df_smoke_models.append(df_smoke_model)

    return df_smoke_models


# Visualizzazione della tabella con le metriche
df_smoke_models_concat = pd.concat(model_report(model), axis=0).reset_index()  # Concatenazione dei modelli
df_smoke_models_concat = df_smoke_models_concat.drop('index', axis=1)  # Rimozione dell'indice
print(df_smoke_models_concat)  # Visualizzazione della tabella

# VERIFICA DELL'IMPORTANZA DELLE FEATURES
# Creazione della feature X e del target y
X = df_smoke.drop('smoking', axis=1)
y = df_smoke['smoking']

# Classificatore da utilizzare per la ricerca delle feature principali
rfc = RandomForestClassifier(random_state=42, n_estimators=100)
rfc_model = rfc.fit(X, y)

# Tracciamento delle feature in base alla loro importanza
plt.style.use('ggplot')
(pd.Series(rfc_model.feature_importances_, index=X.columns)
 .nlargest(10)  # Numero massimo di feature da visualizzare
 .plot(kind='barh', figsize=[10, 5])  # Tipo di grafico e dimensione
 .invert_yaxis())  # Assicuro un ordine decrescente

# Visualizzazione del grafico
plt.title('Top features derived by Random Forest', size=20)
plt.yticks(size=15)
plt.show()

# CREAZIONE DELLA RETE BAYESIANA
# Conversione di tutti i valori all'interno del dataframe in interi
df_smoke_int = np.array(df_smoke, dtype=int)
df_smoke = pd.DataFrame(df_smoke_int, columns=df_smoke.columns)

# Creazione della feature X e del target y
X_train = df_smoke
y_train = df_smoke["smoking"]

# Creazione della struttura della rete
k2 = K2Score(X_train)
hc_k2 = HillClimbSearch(X_train)
k2_model = hc_k2.estimate(scoring_method=k2)

# Creazione della rete
bNet = BayesianNetwork(k2_model.edges())
bNet.fit(df_smoke, estimator=MaximumLikelihoodEstimator)

# Visualizzazione dei nodi e degli archi  - DA TOGLIERE!
print('\033[1m' + '\nNodi della rete:\n' + '\033[0m', bNet.nodes)
print('\033[1m' + '\nArchi della rete:\n' + '\033[0m', bNet.edges)

# CALCOLO DELLA PROBABILITÀ

# Calcolo della probabilità per un soggetto presumibilmente non diabetico (0) ed uno diabetico (1) di avere il diabete

# Eliminazione delle variabili ininfluenti
data = VariableElimination(bNet)

# Soggetto potenzialmente non diabetico
notSmoker = data.query(variables=['smoking'],
                       evidence={'age': 55, 'height(cm)': 170, 'weight(kg)': 60, 'waist(cm)': 8,
                                 'hemoglobin': 158, 'dental caries': 0})

print('\nProbabilità per un soggetto potenzialmente non fumatore:')
print(notSmoker, '\n')

"""
# Soggetto potenzialmente diabetico
smoker = data.query(variables=['smoking'],
                    evidence={'age': 40, 'height(cm)': 155, 'weight(kg)': 45, 'waist(cm)': 590, 'systolic': 950,
                              'relaxation': 520, 'fasting blood sugar': 810, 'Cholesterol': 1550,
                              'triglyceride': 470, 'HDL': 880, 'LDL': 570, 'hemoglobin': 126, 'dental caries': 0})

print('\nProbabilità per un soggetto potenzialmente fumatore:')
print(smoker)
"""
