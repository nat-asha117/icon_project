# Ingegneria della conoscenza
# Natasha Fabrizio - Francesco Saverio Cassano

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pgmpy.estimators import K2Score, HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# Support methods
def prPurple(prt):
    print("\033[95m{}\033[00m".format(prt))


def prRedMoreString(prt, prt2, prt3):
    print("\033[91m{}\033[00m".format(prt), prt2, prt3)


def prGreenMoreString(prt, prt2, prt3):
    print("\n\033[92m{}\033[00m".format(prt), prt2, prt3)


def prRed(prt):
    print("\033[91m{}\033[00m".format(prt))


def prGreen(prt):
    print("\033[92m{}\033[00m".format(prt))


def prYellow(prt):
    print("\033[93m{}\033[00m".format(prt))


def autopct(pct):
    return ('%.2f' % pct + "%") if pct > 1 else ''  # mostra solo i valori delle laber che sono superiori al 1%


df = pd.read_csv("C:\\Users\\natax\\icon_project\\pythonProject\\smoking.csv")

prPurple("\n\n\t\t\tBenvenuto nel nostro sistema per predire se, presi dei soggetti, essi sono fumatori o meno.\n\n")

# DATASET OPTIMIZATION
# Deleting unused and/or irrelevant columns
df_smoke = df.drop(["ID", "gender", "eyesight(left)", "eyesight(right)", "hearing(left)", "hearing(right)", "oral"], axis=1)

# conversione da stringa a intero + pulizia dataframe
df_smoke["tartar"] = df_smoke["tartar"].replace("N", 0)
df_smoke["tartar"] = df_smoke["tartar"].replace("Y", 1)

#df_smoke["height(cm)"] = df_smoke["height(cm)"].astype(float)

# stampa dataframe
print(df_smoke)



# dataset di input eliminando l'ultima colonna in quanto servirà per l'output
X = df_smoke.drop("smoking", axis=1)
Y = df_smoke["smoking"]

# BILANCIAMENTO DELLE CLASSI

# Proporzione dei non fumatori (0) e fumatori (1):
# [Numero di (non) fumatori/Numero totale fumartori]

prGreenMoreString('Non presenza fumo:', df_smoke.smoking.value_counts()[0],
                  '(% {:.2f})'.format(df_smoke.smoking.value_counts()[0] / df_smoke.smoking.count() * 100))
prRedMoreString('Presenza fumo:', df_smoke.smoking.value_counts()[1],
                '(% {:.2f})'.format(df_smoke.smoking.value_counts()[1] / df_smoke.smoking.count() * 100))

# Visualizzazione del grafico

labels = ["Not smokers", "Smokers"]
ax = df_smoke['smoking'].value_counts().plot(kind='pie', figsize=(5, 5), autopct=autopct, labels=None)
ax.axes.get_yaxis().set_visible(False)
plt.title("Graph of occurrence of smokers and non-smokers")
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
bnb = BernoulliNB()
gnb = GaussianNB()

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
            },

    'BernoulliNB': {'accuracy_list': 0.0,
                    'precision_list': 0.0,
                    'recall_list': 0.0,
                    'f1_list': 0.0
                    },

    'GaussianNB': {'accuracy_list': 0.0,
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
    bnb.fit(X_train, y_train)
    gnb.fit(X_train, y_train)

    y_pred_knn = knn.predict(X_test)
    y_pred_dtc = dtc.predict(X_test)
    y_pred_rfc = rfc.predict(X_test)
    y_pred_SVM = svc.predict(X_test)
    y_pred_gnb = gnb.predict(X_test)
    y_pred_bnb = bnb.predict(X_test)

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

    model['BernoulliNB']['accuracy_list'] = (metrics.accuracy_score(y_test, y_pred_bnb))
    model['BernoulliNB']['precision_list'] = (metrics.precision_score(y_test, y_pred_bnb))
    model['BernoulliNB']['recall_list'] = (metrics.recall_score(y_test, y_pred_bnb))
    model['BernoulliNB']['f1_list'] = (metrics.f1_score(y_test, y_pred_bnb))

    model['GaussianNB']['accuracy_list'] = (metrics.accuracy_score(y_test, y_pred_gnb))
    model['GaussianNB']['precision_list'] = (metrics.precision_score(y_test, y_pred_gnb))
    model['GaussianNB']['recall_list'] = (metrics.recall_score(y_test, y_pred_gnb))
    model['GaussianNB']['f1_list'] = (metrics.f1_score(y_test, y_pred_gnb))


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
ax = (pd.Series(rfc_model.feature_importances_, index=X.columns)
      .nlargest(10)  # Numero massimo di feature da visualizzare
      .plot(kind='pie', figsize=(6, 6), autopct=autopct)  # Tipo di grafico e dimensione
      .invert_yaxis())  # Assicuro un ordine decrescente

# Visualizzazione del grafico
plt.title("Top features derived by Random Forest")
plt.ylabel("")
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

# Calcolo della probabilità per un soggetto presumibilmente non fumatore (0) ed uno fumatore (1)

# Eliminazione delle variabili ininfluenti
data = VariableElimination(bNet)

# Soggetto potenzialmente non fumatore
notSmoker = data.query(variables=['smoking'],
                       evidence={'systolic': 102, 'relaxation': 71, 'HDL': 103, 'hemoglobin': 11,
                                 'serum creatinine': 2, 'tartar': 0})

prGreen('\nProbabilità per un soggetto potenzialmente non fumatore:')
print(notSmoker, '\n')

# Test su Soggetto potenzialmente non fumatore
TestnotSmoker = data.query(variables=['smoking'],
                           evidence={'systolic': 81, 'relaxation': 71, 'HDL': 103, 'hemoglobin': 16,
                                     'serum creatinine': 2, 'tartar': 1})

prGreen('\nTest su un soggetto potenzialmente non fumatore:')
print(TestnotSmoker, '\n')

# Soggetto potenzialmente fumatore
smoker = data.query(variables=['smoking'],
                    evidence={'systolic': 93, 'relaxation': 43, 'HDL': 50, 'hemoglobin': 18,
                              'serum creatinine': 5, 'tartar': 1})

prRed('\nProbabilità per un soggetto potenzialmente fumatore:')
print(smoker)

# Test su Soggetto potenzialmente fumatore
TestSmoker = data.query(variables=['smoking'],
                        evidence={'systolic': 75, 'relaxation': 43, 'HDL': 50, 'hemoglobin': 14,
                                  'serum creatinine': 5, 'tartar': 0})

prRed('\nTest su un soggetto potenzialmente fumatore:')
print(TestSmoker, '\n')
