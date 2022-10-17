# Ingegneria della conoscenza
# Natasha Fabrizio - Francesco Saverio Cassano


# Main libraries
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import sys
import time
import multiprocessing

# Classification models
from pgmpy.estimators import K2Score, HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork

# Machine learning
from sklearn import metrics
from sklearn.model_selection import RepeatedKFold, cross_val_score

# Classification algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
import warnings


# Support methods
def simulationThread(bNet, value, data, evemt):
    warnings.filterwarnings("ignore")
    prGreen("Finding combination...")
    time.sleep(1)
    while True:
        newValue = bNet.simulate(show_progress=False, n_samples=1,
                                 evidence={'smoking': 0, 'age': value[0], 'height(cm)': value[1],
                                           'weight(kg)': value[2], 'gender': value[3]})
        UserInputUpdated = data.query(show_progress=False, variables=['smoking'],
                                      evidence={'age': newValue.get("age")[0],
                                                'height(cm)': newValue.get("height(cm)")[0],
                                                'weight(kg)': newValue.get("weight(kg)")[0],
                                                'gender': newValue.get("gender")[0],
                                                'hearing(left)': newValue.get("hearing(left)")[0],
                                                'hearing(right)': newValue.get("hearing(right)")[0],
                                                'triglyceride': newValue.get("triglyceride")[0],
                                                "dental caries": newValue.get("dental caries")[0],
                                                'tartar': newValue.get("tartar")[0],
                                                })
        if UserInputUpdated.values[0] > 0.50:
            time.sleep(1)
            newValue = newValue.drop(
                ["ALT", "waist(cm)", "Gtp", "serum creatinine", "eyesight(right)", "eyesight(left)",
                 "Cholesterol", "LDL", "systolic", "relaxation", "HDL", "hemoglobin"], axis=1)
            prYellow("Suggested values:")
            print(newValue)
            prYellow("New probability based on suggested values")
            print(UserInputUpdated)
            evemt.set()
            break


def prPurple(prt):
    print("\033[95m{}\033[00m".format(prt))


def prRedMoreString(prt, prt2, prt3):
    print("\033[91m{}\033[00m".format(prt), prt2, prt3)


def prYellowMoreString(prt, prt2, prt3):
    print("\033[91m{}\033[00m".format(prt), "\033[91m{}\033[00m".format(prt2), "\033[91m{}\033[00m".format(prt3))


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


def printProbability(data, var):
    print(var, "probability:", data.values[1], "\n")


if __name__ == '__main__':
    # Import of the dataset
    if len(sys.argv) > 1:
        df = pd.read_csv(sys.argv[1])
    else:
        df = pd.read_csv("C:\\Users\\verio\\repo\\progettoICON\\pythonProject\\smoking.csv")

    # DATASET OPTIMIZATION:

    # Deleting unused and/or irrelevant columns
    df_smoke = df.drop(["ID"], axis=1)

    # String to full conversion + dataframe cleaning
    df_smoke["tartar"] = df_smoke["tartar"].replace("N", 0)
    df_smoke["tartar"] = df_smoke["tartar"].replace("Y", 1)
    df_smoke["oral"] = df_smoke["oral"].replace("N", 0)
    df_smoke["oral"] = df_smoke["oral"].replace("Y", 1)
    df_smoke["gender"] = df_smoke["gender"].replace("M", 0)
    df_smoke["gender"] = df_smoke["gender"].replace("F", 1)

    # Data overview
    print("\nDisplay (partial) of the dataframe:\n", df_smoke.head())
    print("\nNumber of elements: ", len(df_smoke.index) - 1)
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
    prGreenMoreString('Not smokers: ', df_smoke.smoking.value_counts()[0],
                      '(% {:.2f})'.format(df_smoke.smoking.value_counts()[0] / df_smoke.smoking.count() * 100))
    prRedMoreString('Smokers: ', df_smoke.smoking.value_counts()[1],
                    '(% {:.2f})'.format(df_smoke.smoking.value_counts()[1] / df_smoke.smoking.count() * 100))

    df_majority = df_smoke[df_smoke["smoking"] == 0]
    df_minority = df_smoke[df_smoke["smoking"] == 1]
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=4473, random_state=42)
    df_smoke = pd.concat([df_minority_upsampled, df_majority])

    prYellow("\nValue after Oversampling:")
    prGreenMoreString('Not smokers: ', df_smoke.smoking.value_counts()[0],
                      '(% {:.2f})'.format(df_smoke.smoking.value_counts()[0] / df_smoke.smoking.count() * 100))
    prRedMoreString('Smokers: ', df_smoke.smoking.value_counts()[1],
                    '(% {:.2f})'.format(df_smoke.smoking.value_counts()[1] / df_smoke.smoking.count() * 100))

    # Visualization of the aspect ratio chart
    labels = ["Not smokers", "Smokers"]
    ax = df_smoke['smoking'].value_counts().plot(kind='pie', figsize=(5.7, 5.7), autopct=autopct, labels=None)
    ax.axes.get_yaxis().set_visible(False)
    plt.title("Graph of occurrence of smokers and non-smokers\n\nafter Oversampling")
    plt.legend(labels=labels, loc="best")
    plt.show()

    # EVALUATION SELECTION: K-FOLD CROSS VALIDATION

    # Creation of X feature and target y
    X = df_smoke.to_numpy()
    y = df_smoke["smoking"].to_numpy()  # K-Fold Cross Validation

    kf = RepeatedKFold(n_splits=5, n_repeats=5)

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
                df_smoke_model = pd.DataFrame({'model': [clf],
                                               'accuracy': [np.mean(model[clf]['accuracy_list'])],
                                               'precision': [np.mean(model[clf]['precision_list'])],
                                               'recall': [np.mean(model[clf]['recall_list'])],
                                               'f1score': [np.mean(model[clf]['f1_list'])]
                                               })

                df_smoke_models.append(df_smoke_model)
            return df_smoke_models

    # Visualization of the table with metrics and Graph
    df_smoke_models_concat = pd.concat(model_report(model), axis=0).reset_index()  # concatenation of the models
    df_smoke_models_concat = df_smoke_models_concat.drop('index', axis=1)  # removal of the index
    print("\n", df_smoke_models_concat)  # table display

    # Accuracy Graph
    x = df_smoke_models_concat.model
    y = df_smoke_models_concat.accuracy

    plt.bar(x, y)
    plt.title("Accuracy")
    plt.show()

    # Precision Graph
    x = df_smoke_models_concat.model
    y = df_smoke_models_concat.precision

    plt.bar(x, y)
    plt.title("Precision")
    plt.show()

    # Recall Graph
    x = df_smoke_models_concat.model
    y = df_smoke_models_concat.recall

    plt.bar(x, y)
    plt.title("Recall")
    plt.show()

    # F1score Graph
    x = df_smoke_models_concat.model
    y = df_smoke_models_concat.f1score

    plt.bar(x, y)
    plt.title("F1score")
    plt.show()

    # Standard deviation
    std_knn = np.std(cross_val_score(knn, X_test, y_test, cv=5, n_jobs=5))
    std_dtc = np.std(cross_val_score(dtc, X_test, y_test, cv=5, n_jobs=5))
    std_rfc = np.std(cross_val_score(rfc, X_test, y_test, cv=5, n_jobs=5))
    std_svc = np.std(cross_val_score(svc, X_test, y_test, cv=5, n_jobs=5))
    std_bnd = np.std(cross_val_score(bnb, X_test, y_test, cv=5, n_jobs=5))
    std_gnb = np.std(cross_val_score(gnb, X_test, y_test, cv=5, n_jobs=5))
    plt.plot(["KNN", "DecisionTree", "RandomForest", "SVM", "BernoulliNB", "GaussianNB"],
             [std_knn, std_dtc, std_rfc, std_svc, std_bnd, std_gnb])
    plt.title("Standard deviation")
    plt.ylabel("Standard deviation value")
    plt.xlabel("Classifiers")
    plt.show()
    print("\nStandard deviation for Knn:", std_knn)
    print("\nStandard deviation for DecisionTree:", std_dtc)
    print("\nStandard deviation for RandomForest:", std_rfc)
    print("\nStandard deviation for SVM:", std_svc)
    print("\nStandard deviation for BernoulliNB:", std_bnd)
    print("\nStandard deviation for GaussianNB:", std_gnb)
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

    prYellow("\n\t\tCreation of the Bayesian Network\n")

    # Removing unnecessary features
    df_smoke = df_smoke.drop(["oral", "Cholesterol", "LDL"], axis=1)

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

    # Graph of nodes

    G = nx.MultiDiGraph(k2_model.edges())
    G.add_edges_from(bNet.edges())
    pos = nx.spring_layout(G, iterations=100, k=2, threshold=5, pos=nx.spiral_layout(G))
    nx.draw_networkx_nodes(G, pos, node_size=150, node_color="#ff574c")
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", clip_on=True, horizontalalignment="center",
                            verticalalignment="bottom")
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=7, arrowstyle="->", edge_color="purple",
                           connectionstyle="angle3,angleA=90,angleB=0", min_source_margin=1.2, min_target_margin=1.5,
                           edge_vmin=2, edge_vmax=2)

    plt.title("BAYESIAN NETWORK GRAPH")
    plt.show()

    # Information about bNet

    prYellow("\nMarkov blanket for \"smoking\"")
    print(bNet.get_markov_blanket('smoking'), "\n")

    prYellow("\nMarkov blanket for \"gender\"")
    print(bNet.get_markov_blanket('gender'), "\n")

    # CALCULATION OF THE PROBABILITY
    #  calculation for a supposed non-smoker (0) and a smoker (1)

    # Elimination of irrelevant variables
    data = VariableElimination(bNet)  # inference

    prGreen("Probability control of features that affect gender:")
    prGreen("Test on a non-smoker female subject")
    woman = data.query(show_progress=False, variables=['gender'], evidence={'age': 40, 'height(cm)': 160,
                                                                            'weight(kg)': 65, 'Gtp': 34, 'HDL': 55,
                                                                            'serum creatinine': 0, 'systolic': 126,
                                                                            'ALT': 15, 'eyesight(left)': 1,
                                                                            'eyesight(right)': 0, 'hemoglobin': 13,
                                                                            'smoking': 0})
    print(woman, '\n')
    prGreen("Test on a non-smoker male subject")
    man = data.query(show_progress=False, variables=['gender'], evidence={'age': 75, 'height(cm)': 160,
                                                                          'weight(kg)': 70, 'Gtp': 17, 'HDL': 71,
                                                                          'serum creatinine': 0, 'systolic': 128,
                                                                          'ALT': 16,
                                                                          'eyesight(left)': 0, 'eyesight(right)': 1,
                                                                          'hemoglobin': 15, 'smoking': 0})
    print(man, '\n')

    # Smoker

    prRed("Test on a smoker female subject")
    woman = data.query(show_progress=False, variables=['gender'], evidence={'age': 40, 'height(cm)': 160,
                                                                            'weight(kg)': 65, 'Gtp': 34, 'HDL': 55,
                                                                            'serum creatinine': 0, 'systolic': 126,
                                                                            'ALT': 15,
                                                                            'eyesight(left)': 1, 'eyesight(right)': 0,
                                                                            'hemoglobin': 13, 'smoking': 1})
    print(woman, '\n')
    prRed("Test on a smoker male subject")
    man = data.query(show_progress=False, variables=['gender'], evidence={'age': 75, 'height(cm)': 160,
                                                                          'weight(kg)': 70, 'Gtp': 17, 'HDL': 71,
                                                                          'serum creatinine': 0, 'systolic': 128,
                                                                          'ALT': 16,
                                                                          'eyesight(left)': 0, 'eyesight(right)': 1,
                                                                          'hemoglobin': 15, 'smoking': 1})
    print(man, '\n')

    # Potential non-smoker subject
    prGreen("Tests carried out on an average person with values:")
    print("age: 20\t-\theight(cm): 170\t-\tweight(kg): 60\t-\tgender: 1 (woman)\n")
    notSmoker = data.query(show_progress=False, variables=['smoking'],
                           evidence={'age': 20, 'height(cm)': 170, 'weight(kg)': 60, 'gender': 1, 'hearing(left)': 0,
                                     'hearing(right)': 0, 'triglyceride': 113, 'dental caries': 0, 'tartar': 0})

    prGreen('\nProbability for a potentially non-smoker:')
    print(notSmoker, '\n')

    # Test on Potentially non-smoker subject
    TestNotSmoker = data.query(show_progress=False, variables=['smoking'],
                               evidence={'age': 20, 'height(cm)': 170, 'weight(kg)': 60, 'gender': 1,
                                         'triglyceride': 148, 'hearing(left)': 1, 'hearing(right)': 1,
                                         'dental caries': 1, 'tartar': 1})

    prGreen('\nTest on Potentially non-smoker subject:')
    print(TestNotSmoker, '\n')

    # Potential smoker
    smoker = data.query(show_progress=False, variables=['smoking'],
                        evidence={'age': 20, 'height(cm)': 170, 'weight(kg)': 60, 'gender': 1, 'triglyceride': 151,
                                  'hearing(left)': 1, 'hearing(right)': 1, 'dental caries': 1, 'tartar': 1})

    prRed('\nProbability for a potential smoker:')
    print(smoker)

    # Test on subject potentially smoker
    TestSmoker = data.query(show_progress=False, variables=['smoking'],
                            evidence={'age': 20, 'height(cm)': 170, 'weight(kg)': 60, 'gender': 1, 'triglyceride': 120,
                                      'hearing(left)': 1, 'hearing(right)': 1, 'dental caries': 0, 'tartar': 0})

    prRed('\nTest on Subject potentially smoker:')
    print(TestSmoker, '\n')

    prYellow("\n\n\t\t\t\t\tWelcome to our system!\n\n\t"
             "It allows you to predict whether, taken of the subjects, they are smokers or not.\n\n")

    #
    while True:
        i = 0
        try:
            prYellow("Do you want to enter your data for a prediction? - Y/N? - (Typing 'n' close program)")
            result = str(input())
            if 'N' == result or result == 'n':
                exit(1)
            elif 'Y' == result or result == 'y':
                prYellow("Please insert: ")
                columns = ["age", "height(cm)", "weight(kg)", "gender", "hearing(left)", "hearing(right)",
                           "triglyceride",
                           "dental caries", "tartar"]
                print(columns)
                prRed("Age - height(cm) - weight(kg) - gender are obligatory to enter!")
                value = [None] * len(columns)
                while i < len(columns):
                    if columns[i] == "age" or columns[i] == "height(cm)" or columns[i] == "weight(kg)":
                        prRed("The range of allowed values are multiples of 5")
                        print("The minimum acceptable \"", columns[i], "\" value is:", df_smoke[columns[i]].min(),
                              "The maximum is:", df_smoke[columns[i]].max())
                        print("Insert ", columns[i], " value: ")
                    elif columns[i] != "tartar" and columns[i] != "dental caries" and columns[i] != "gender":
                        print("Insert ", columns[i], " value (if you don’t have the value, enter -1): ")
                    elif columns[i] == "gender":
                        print("Insert ", columns[i], " value (0 = Man, 1 = Woman)")
                    elif (columns[i] == "hearing(left)") or (columns[i] == "hearing(right)"):
                        print("Insert ", columns[i], " value (Acceptable values range from 0 to 2)")
                    else:
                        print("Insert ", columns[i], " value (0 = No, 1 = Yes, -1 = Data not available): ")
                    value[i] = int(input())
                    if columns[i] == "age" or columns[i] == "height(cm)" or columns[i] == "weight(kg)":
                        if value[i] <= 0:
                            prRed("Insert value >= 0")
                        elif value[i] < df_smoke[columns[i]].min():
                            prRed("Error! You entered too small value!")
                        elif value[i] > df_smoke[columns[i]].max():
                            prRed("Error! You entered too large value!")
                        elif value[i] % 5 != 0:
                            prRed("Error! You have not entered a multiple of 5")
                        else:
                            i = i + 1
                    elif columns[i] == "gender":
                        if value[i] < 0:
                            prRed("Insert value >= 0")
                        elif value[i] < df_smoke[columns[i]].min():
                            prRed("Error! You entered too small value!")
                        elif value[i] > df_smoke[columns[i]].max():
                            prRed("Error! You entered too large value!")
                        else:
                            i = i + 1
                    else:
                        if value[i] == -1:
                            prRed("Insert value >= 0")
                        elif value[i] <= -2:
                            prRed("Insert value >= 0")
                        elif (columns[i] == "tartar") and (value[i] > 1):
                            prRed("Error! Insert value (0 = No, 1 = Yes): ")
                        elif (columns[i] == "dental caries") and (value[i] > 1):
                            prRed("Error! Insert value (0 = No, 1 = Yes): ")
                        elif (columns[i] == "hearing(left)" or columns[i] == "hearing(right)") and (
                                value[i] > 2 or value[i] < 0):
                            prRed("Error! Insert value in range 0 - 2")
                        else:
                            i = i + 1
                try:
                    i = 0
                    dataAvailable = {}
                    while i < len(columns):
                        if value[i] != -1:
                            dataAvailable[columns[i]] = value[i]
                        i = i + 1
                    UserInput = data.query(show_progress=False, variables=['smoking'],
                                           evidence=dataAvailable)
                    print(UserInput)
                    if UserInput.values[0] <= 0.50:
                        prYellow("Want to know what values to improve not to be to no longer be considered a smoker?"
                                 " - Y/N")
                        result = str(input())
                        if 'Y' == result or result == 'y':
                            waitTime = float(60)
                            prYellowMoreString("The search will last", waitTime, "seconds. If no ideal combinations "
                                                                                 "are found, no data will be "
                                                                                 "displayed.")
                            event = multiprocessing.Event()
                            t = multiprocessing.Process(target=simulationThread, args=(bNet, value, data, event))
                            t.start()
                            if not event.wait(waitTime):
                                prRed("Error! No data find.")
                            event.clear()
                            t.terminate()
                except IndexError as e:
                    prRed("Error!")
                    print("You are insert:  ", value)
                    print(e.args)
            else:
                print("Wrong input. Write Y or N")
        except ValueError:
            print("Wrong input")
