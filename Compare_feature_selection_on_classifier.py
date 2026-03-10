from utilities import *
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


filePath = "./ObesityDataSet_raw_and_data_sinthetic.csv"
df = pd.read_csv(filePath)
pcaMode = False
rfeMode = True

binary_cols = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
df[binary_cols] = df[binary_cols].apply(lambda x: x.map({'yes': 1, 'no': 0}))
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
freq_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
df['CAEC'] = df['CAEC'].map(freq_map)
df['CALC'] = df['CALC'].map(freq_map)
df = pd.get_dummies(df, columns=['MTRANS'])

X = df.drop(columns=['NObeyesdad'])
le = LabelEncoder()
y = le.fit_transform(df['NObeyesdad'])
classNames = le.classes_
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
print(f"Split done — Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

X_trainScaled, X_trainMean, X_trainStd = standarizationCal(X_train, mode="train", trainMean=None, 
                                                           trainStd=None)
X_testScaled = standarizationCal(X_test, mode="test", trainMean=X_trainMean, trainStd=X_trainStd)
X_valScaled = standarizationCal(X_val, mode="test", trainMean=X_trainMean, trainStd=X_trainStd)
print("Standardization done")

if pcaMode:
    X_trainPCA, eigenVecTopFromTrain = pcaCal(X_trainScaled, nComponents=10, mode="train", 
                                              eigenVecTop=None)
    X_testPCA = pcaCal(X_testScaled, nComponents=2, mode="test", eigenVecTop=eigenVecTopFromTrain)
    X_valPCA = pcaCal(X_valScaled, nComponents=2, mode="test", eigenVecTop=eigenVecTopFromTrain)
    print("PCA done — Train:", X_trainPCA.shape)

    kChoice = kSelection(X_train=X_trainPCA, y_train=y_train, X_val=X_valPCA, y_val=y_val)
    print(f"Best k selected: {kChoice}")
    y_predict = knnCal(X_train=X_trainPCA, y_train=y_train, X_test=X_testPCA, k=kChoice)

if rfeMode:
    estimator = LogisticRegression(max_iter=1000)
    rfe = RFE(estimator, n_features_to_select=10)   
    X_trainRFE = rfe.fit_transform(X_trainScaled, y_train)
    X_valRFE = rfe.transform(X_valScaled)
    X_testRFE = rfe.transform(X_testScaled)
    print("RFE done — Train:", X_trainRFE.shape)

    kChoice = kSelection(X_train=X_trainRFE, y_train=y_train, X_val=X_valRFE, y_val=y_val)
    print(f"Best k selected: {kChoice}")
    y_predict = knnCal(X_train=X_trainRFE, y_train=y_train, X_test=X_testRFE, k=kChoice)


confusionMatrix = confusionMatrixGen(y_true=y_test, y_prediction=y_predict)
confusionMatrixVisual = visualizeCM(confusionMatrix=confusionMatrix, classes=classNames)