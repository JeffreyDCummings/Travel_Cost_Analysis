"""Trains an XGBoost Classifier model of optmized hyperparameters, and shows its validation
accuracy through a stratified KFold validation with 20 splits, reaching ~63%, which is a
successful accuracy for a 5-class problem."""
import ast
from statistics import mean
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

CLASS_NUM = 5

def classifier(toll_cost_tags):
    """Converts the toll cost tags output for each route into a cost class. """
    y_out = []
    count = [0 for _ in range(CLASS_NUM)]
    for cost in toll_cost_tags:
        if cost == 0:
            y_out.append(0)
            count[0] += 1
        elif cost <= 8:
            y_out.append(1)
            count[1] += 1
        elif cost < 23:
            y_out.append(2)
            count[2] += 1
        elif cost < 60:
            y_out.append(3)
            count[3] += 1
        else:
            y_out.append(4)
            count[4] += 1
    return y_out, count

def onehotencodedtolls(ml_data_train):
    """Manually one hot encodes the list of tolls that each trips fastest route passes through.
    If there are no tolls, a dummy toll of ID = 0 is encoded instead.  This also replaces the
    need for a boolean feature of whether or not the route has tolls."""
    toll_ids = ml_data_train["Toll IDs"].fillna("[0]").tolist()

    id_set = set()
    for id_num in toll_ids:
        toll_list = ast.literal_eval(id_num)
        for id_int in toll_list:
            id_set.add(id_int)

    one_hot_encoded_tolls = pd.DataFrame(0, index=np.arange(len(toll_ids)), columns=id_set)

    for route, id_num in enumerate(toll_ids):
        toll_list = ast.literal_eval(id_num)
        for id_int in toll_list:
            one_hot_encoded_tolls[id_int][route] = 1
    return one_hot_encoded_tolls

def feature_engineering():
    """Extracts the features from the file and converts them into the necessary format to
    read into the ML model."""
    ml_data_train = pd.read_csv("ml_data_zero.csv", delimiter=",")
    x_train = ml_data_train[["Start Latitude", "Start Longitude", "Stop Latitude",\
     "Stop Longitude", "Distance", "Duration"]].copy()
    y_train = ml_data_train["Min Rate Tags"].copy()
    y_train, train_count = classifier(y_train)
    print("Class Sizes: ", train_count)
    one_hot_encoded_tolls = onehotencodedtolls(ml_data_train)
    x_train = pd.concat([x_train, one_hot_encoded_tolls], axis=1)
    return x_train.to_numpy(), np.array(y_train)

def kfold_validation(x_train, y_train):
    """Validates the model on the training data through a stratified KFold of 20 splits."""
    statistics = []
    kfold = StratifiedKFold(n_splits=20, shuffle=True)
    for train, test in kfold.split(x_train, y_train):
        smote_app = SMOTE(sampling_strategy='auto')
        data_smote_train, target_smote_train = smote_app.fit_sample(x_train[train], y_train[train])
        model = XGBClassifier(learning_rate=0.05, n_estimators=110, max_depth=11,\
         min_child_weight=0, gamma=0.2, subsample=0.6, colsample_bytree=0.2, reg_alpha=1e-03,\
         objective='multi:softmax', nthread=4, num_class=CLASS_NUM, scale_pos_weight=1)
        model.fit(data_smote_train, target_smote_train)
        y_prediction = model.predict(x_train[test])
        print("Accuracy: ", round(metrics.accuracy_score(y_train[test], y_prediction), 3))
        statistics.append(metrics.accuracy_score(y_train[test], y_prediction))
    print("Mean KFold Accuracy: ", mean(statistics))

def main():
    """The main program that runs the necessary functions."""
    x_train, y_train = feature_engineering()
    kfold_validation(x_train, y_train)

if __name__ == "__main__":
    main()
