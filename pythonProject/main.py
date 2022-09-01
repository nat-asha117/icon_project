# Ingegneria della conoscenza
# Natasha Fabrizio - Francesco Saverio Cassano

# Main libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Classification models
from pgmpy.estimators import K2Score, HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork

# Machine learning
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

# Classification algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample


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
    return ('%.2f' % pct + "%") if pct > 1 else ''  # shows only values of labers that are greater than 1%


# Import of the dataset
df = pd.read_csv("C:\\Users\\natax\\icon_project\\pythonProject\\smoking.csv")

prYellow("\n\n\t\t\t\t\t\t\t\tWelcome to our system!\n\n\t"
         "It allows you to predict whether, taken of the subjects, they are smokers or not.\n\n")

# DATASET OPTIMIZATION:

# Deleting unused and/or irrelevant columns
df_smoke = df.drop(["ID", "gender", "eyesight(left)", "eyesight(right)", "hearing(left)", "hearing(right)", "oral"], axis=1)

# String to full conversion + dataframe cleaning
df_smoke["tartar"] = df_smoke["tartar"].replace("N", 0)
df_smoke["tartar"] = df_smoke["tartar"].replace("Y", 1)

# Data overview
print("\nDisplay (partial) of the dataframe:\n", df_smoke.head())
print("\nInfo dataset:\n", df_smoke.describe())
# Input dataset, eliminating the last column (needed for the output)
X = df_smoke.drop("smoking", axis=1)
Y = df_smoke["smoking"]


# BALANCING OF CLASSES

# Visualization of the aspect ratio chart
labels = ["Not smokers", "Smokers"]
ax = df_smoke['smoking'].value_counts().plot(kind='pie', figsize=(5, 5), autopct=autopct, labels=None)
ax.axes.get_yaxis().set_visible(False)
plt.title("Graph of occurrence of smokers and non-smokers")
plt.legend(labels=labels, loc="best")
plt.show()

# Proportion of non-smokers (0) and smokers (1):
# [Number of non-smokers/Total number of smokers]
prGreenMoreString('Smoke-free:', df_smoke.smoking.value_counts()[0],
                  '(% {:.2f})'.format(df_smoke.smoking.value_counts()[0] / df_smoke.smoking.count() * 100))
prRedMoreString('Presence of smoke:', df_smoke.smoking.value_counts()[1],
                '(% {:.2f})'.format(df_smoke.smoking.value_counts()[1] / df_smoke.smoking.count() * 100))

df_majority = df_smoke[df_smoke["smoking"]==0]
df_minority = df_smoke[df_smoke["smoking"]==1]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=4473, random_state=42)
df_smoke = pd.concat([df_minority_upsampled, df_majority])

prYellow("\nValue after Oversampling:")
prGreenMoreString('Smoke-free:', df_smoke.smoking.value_counts()[0],
                  '(% {:.2f})'.format(df_smoke.smoking.value_counts()[0] / df_smoke.smoking.count() * 100))
prRedMoreString('Presence of smoke:', df_smoke.smoking.value_counts()[1],
                '(% {:.2f})'.format(df_smoke.smoking.value_counts()[1] / df_smoke.smoking.count() * 100))


# Visualization of the aspect ratio chart
labels = ["Not smokers", "Smokers"]
ax = df_smoke['smoking'].value_counts().plot(kind='pie', figsize=(5, 5), autopct=autopct, labels=None)
ax.axes.get_yaxis().set_visible(False)
plt.title("Graph of occurrence of smokers and non-smokers\n\nafter Oversampling")
plt.legend(labels=labels, loc="best")
plt.show()

# EVALUATION SELECTION: K-FOLD CROSS VALIDATION

# Creation of X feature and target y
X = df_smoke.to_numpy()
y = df_smoke["smoking"].to_numpy()  # K-Fold Cross Validation

kf = StratifiedKFold(n_splits=5)  # used because the class is unbalanced

# Classifiers for the purpose of evaluation
knn = KNeighborsClassifier()
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
svc = SVC()
bnb = BernoulliNB()
gnb = GaussianNB()

# Score of metrics
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

# K-Fold of the classifiers
for train_index, test_index in kf.split(X, y):
    training_set, testing_set = X[train_index], X[test_index]

    # train data
    data_train = pd.DataFrame(training_set, columns=df_smoke.columns)
    X_train = data_train.drop("smoking", axis=1)
    y_train = data_train.smoking

    # test data
    data_test = pd.DataFrame(testing_set, columns=df_smoke.columns)
    X_test = data_test.drop("smoking", axis=1)
    y_test = data_test.smoking

    # classifier fit
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

    # saving fold metrics in the dictionary
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


    # report template
    def model_report(model):

        df_smoke_models = []

        for clf in model:
         df_smoke_model = pd.DataFrame({'model'     : [clf],
                                        'accuracy'  : [np.mean(model[clf]['accuracy_list'])],
                                        'precision' : [np.mean(model[clf]['precision_list'])],
                                        'recall'    : [np.mean(model[clf]['recall_list'])],
                                        'f1score'   : [np.mean(model[clf]['f1_list'])]
                                        })

         df_smoke_models.append(df_smoke_model)

        return df_smoke_models

# Visualization of the table with metrics
df_smoke_models_concat = pd.concat(model_report(model), axis=0).reset_index()  # concatenation of the models
df_smoke_models_concat = df_smoke_models_concat.drop('index', axis=1)  # removal of the index
print("\n", df_smoke_models_concat)  # table display


# VERIFICATION OF THE IMPORTANCE OF FEATURES

# Creation of X feature and target y
X = df_smoke.drop('smoking', axis=1)
y = df_smoke['smoking']

# Classifier to be used for the search of the main features
rfc = RandomForestClassifier(random_state=42, n_estimators=100)
rfc_model = rfc.fit(X, y)

# Tracking features based on their importance
ax = (pd.Series(rfc_model.feature_importances_, index=X.columns)
      .nlargest(10)  # maximum number of features to display
      .plot(kind='pie', figsize=(6, 6), autopct=autopct)  # type of chart and size
      .invert_yaxis())  # to ensure a descending order

# Visualization of the graph of the most important features
plt.title("Top features derived by Random Forest")
plt.ylabel("")
plt.show()


# CREATION OF THE BAYESIAN NETWORK

# Converting all values within the dataframe to integers
df_smoke_int = np.array(df_smoke, dtype=int)
df_smoke = pd.DataFrame(df_smoke_int, columns=df_smoke.columns)

# Creation of X feature and target y
X_train = df_smoke
y_train = df_smoke["smoking"]

# Creation of the network structure
k2 = K2Score(X_train)
hc_k2 = HillClimbSearch(X_train)
k2_model = hc_k2.estimate(scoring_method=k2)

# Creation of the network
bNet = BayesianNetwork(k2_model.edges())
bNet.fit(df_smoke, estimator=MaximumLikelihoodEstimator)


# CALCULATION OF THE PROBABILITY
# Probability calculation for a supposedly non-smoker (0) and a smoker (1)

# Elimination of irrelevant variables
data = VariableElimination(bNet)  # inference

# Potential non-smoker subject
notSmoker = data.query(variables=['smoking'],
                       evidence={'Gtp': 31, 'triglyceride': 113, 'LDL': 116, 'systolic': 102,  'relaxation': 71,
                                 'HDL': 103, 'hemoglobin': 13, 'serum creatinine': 2, 'tartar': 0})

prGreen('\nProbability for a potentially non-smoker:')
print(notSmoker, '\n')

# Test on Potentially non-smoker subject
TestnotSmoker = data.query(variables=['smoking'],
                           evidence={'Gtp': 53, 'triglyceride': 148, 'LDL': 116, 'systolic': 102, 'relaxation': 71,
                                     'HDL': 103, 'hemoglobin': 17, 'serum creatinine': 2, 'tartar': 1})

prGreen('\nTest on Potentially non-smoker subject:')
print(TestnotSmoker, '\n')

# Potential smoker
smoker = data.query(variables=['smoking'],
                    evidence={'Gtp': 55, 'triglyceride': 151, 'LDL': 113, 'systolic': 93, 'relaxation': 43, 'HDL': 50,
                              'hemoglobin': 18, 'serum creatinine': 5, 'tartar': 1})

prRed('\nProbability for a potential smoker:')
print(smoker)

# Test on subject potentially smoker
TestSmoker = data.query(variables=['smoking'],
                        evidence={'Gtp': 32, 'triglyceride': 111, 'LDL': 113, 'systolic': 93, 'relaxation': 43,
                                  'HDL': 50, 'hemoglobin': 13, 'serum creatinine': 5, 'tartar': 0})

prRed('\nTest on Subject potentially smoker:')
print(TestSmoker, '\n')
