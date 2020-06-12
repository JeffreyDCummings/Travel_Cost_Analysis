"""Trains an XGBoost, MLP, and Random Forest Classifier model with optmized hyperparameters, 
and validates their accuracies through a stratified KFold validation with 20 splits, reaching ~63%
with each.  Then it builds a stacked classifier with XGBoost on top of these 3 submodels, reaching
66% accuracy."""
import ast
import os
from statistics import mean
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import ordinal_categorical_crossentropy as OCC

#This suppresses multiple tensorflow warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

def multi_class(cost_classes):
    """One hot encodes the cost classes, which is needed for building the Keras model. """
    y_out_class = []
    for cost in cost_classes:
        y_out_class.append([1 if index == cost else 0 for index in range(CLASS_NUM)])
    return np.array(y_out_class)

def onehotencodedtolls(ml_data_train):
    """Manually one hot encodes the list of tolls that each trip's fastest route passes through.
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

def mlp_model(data_train_norm, target_train_hot, data_test_norm, target_test_hot):
    """ Definition of the sequential MLP model within Keras """
    model = Sequential()
    model.add(Dense(232, input_dim=232))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(115))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss=OCC.loss, optimizer='adam', metrics=['accuracy'])
    model.fit(data_train_norm, target_train_hot, epochs=17, validation_data=\
     [data_test_norm, target_test_hot], verbose=0)
    loss, acc = model.evaluate(data_test_norm, target_test_hot)
    return model.predict_proba(data_test_norm), acc, loss

def xgboost(data_train, target_train, data_test, target_test):
    """ Build the XGBoost submodel. """
    model = XGBClassifier(learning_rate=0.05, n_estimators=110, max_depth=11,\
     min_child_weight=0, gamma=0.2, subsample=0.6, colsample_bytree=0.2, reg_alpha=1e-03,\
     objective='multi:softmax', nthread=4, num_class=CLASS_NUM, scale_pos_weight=1)
    model.fit(data_train, target_train)
    y_prediction = model.predict(data_test)
    return model.predict_proba(data_test), metrics.accuracy_score(target_test, y_prediction)

def rf_model(data, target, data_test, target_test):
    """ Build the random forest submodel. """
    rf_clf = RandomForestClassifier(n_estimators=120, min_samples_split=2, min_samples_leaf=1)
    rf_clf = rf_clf.fit(data, target)
    y_prediction = rf_clf.predict(data_test)
    return rf_clf.predict_proba(data_test), metrics.accuracy_score(target_test, y_prediction)

def ensemble_class(ensemble, target, ensemble_test, target_test):
    """ The stacked ensemble model, which is simply an XGB classifier with no tuning. """
    ensemble_clf = XGBClassifier()
    ensemble_clf.fit(ensemble, target)
    y_prediction = ensemble_clf.predict(ensemble_test)
    acc_out = metrics.accuracy_score(target_test, y_prediction)
    return acc_out

def kfold_validation(x_train, y_train):
    """Validates the model on the training data through a stratified KFold of 20 splits."""
    mlp_prob_tot, xgb_prob_tot, rf_prob_tot, target_tot = [], [], [], []
    mlp_acc_tot, xgb_acc_tot, rf_acc_tot, mlp_loss_tot = [], [], [], []
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    for _ in range(2):
        kfold = StratifiedKFold(n_splits=20, shuffle=True)
        for train, test in kfold.split(x_train, y_train):
            data_train_norm = scaler.transform(x_train[train])
            data_test_norm = scaler.transform(x_train[test])
            target_train_hot = multi_class(y_train[train])
            target_test_hot = multi_class(y_train[test])
            mlp_prob, mlp_acc, mlp_loss = mlp_model(data_train_norm, target_train_hot,\
             data_test_norm, target_test_hot)
            xgb_prob, xgb_acc = xgboost(x_train[train], y_train[train], x_train[test],\
             y_train[test])
            rf_prob, rf_acc = rf_model(x_train[train], y_train[train], x_train[test],\
             y_train[test])
            mlp_prob_tot.extend(mlp_prob)
            xgb_prob_tot.extend(xgb_prob)
            rf_prob_tot.extend(rf_prob)
            mlp_acc_tot.append(mlp_acc)
            mlp_loss_tot.append(mlp_loss)
            target_tot.extend(y_train[test])
            xgb_acc_tot.append(xgb_acc)
            rf_acc_tot.append(rf_acc)

    target_tot = np.asarray(target_tot)
    ensemble_prob_tot = np.column_stack([mlp_prob_tot, xgb_prob_tot, rf_prob_tot])
    out = []
    for _ in range(2):
        kfold = StratifiedKFold(n_splits=20, shuffle=True)
        for train, test in kfold.split(ensemble_prob_tot, target_tot):
            acc_out = ensemble_class(ensemble_prob_tot[train], target_tot[train],\
             ensemble_prob_tot[test], target_tot[test])
            out.append(acc_out)

    print("MLP KFold Accuracy, loss: ", mean(mlp_acc_tot), mean(mlp_loss_tot))
    print("XGB KFold Accuracy: ", mean(xgb_acc_tot))
    print("RF KFold Accuracy: ", mean(rf_acc_tot))
    print("Ensemble KFold Accuracy: ", mean(out))

def main():
    """The main program that runs the necessary functions."""
    x_train, y_train = feature_engineering()
    kfold_validation(x_train, y_train)

if __name__ == "__main__":
    main()
