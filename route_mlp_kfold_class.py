"""Trains an XGBoost Classifier model of optmized hyperparameters, and shows its validation
accuracy through a stratified KFold validation with 20 splits, reaching ~63%, which is a
successful accuracy for a 5-class problem."""
import ast
import os
from statistics import mean
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

#This suppresses multiple tensorflow warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CLASS_NUM = 5
EPOCHS = 16

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

def class_ordinal(cost_classes):
    """Converts the toll cost tags output for each route into a cost class. """
    y_out_ord = []
    for cost in cost_classes:
        if cost == 0:
            y_out_ord.append([0, 0, 0, 0])
        elif cost == 1:
            y_out_ord.append([1, 0, 0, 0])
        elif cost == 2:
            y_out_ord.append([1, 1, 0, 0])
        elif cost == 3:
            y_out_ord.append([1, 1, 1, 0])
        else:
            y_out_ord.append([1, 1, 1, 1])
    return np.array(y_out_ord)

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

def mlp_model(data_train_norm, target_train_ord, data_test_norm, target_test_ord):
    """ Definition and fitting of the sequentle MLP model within Keras """
    model = Sequential()
    model.add(Dense(232, input_dim=232))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(115))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(4, activation='sigmoid'))
    y_integers = np.sum(target_train_ord, axis=1)
    class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
    d_class_weights = dict(enumerate(class_weights))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(data_train_norm, target_train_ord, class_weight=d_class_weights,\
     epochs=EPOCHS, verbose=0)
    loss, acc = model.evaluate(data_test_norm, target_test_ord)
    return acc, loss

def kfold_validation(x_train, y_train):
    """Validates the MLP model on the training data through a stratified KFold of 20 splits."""
    mlp_acc_tot, mlp_loss_tot = [], []
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    for _ in range(2):
        kfold = StratifiedKFold(n_splits=20, shuffle=True)
        for train, test in kfold.split(x_train, y_train):
            data_train_norm = scaler.transform(x_train[train])
            data_test_norm = scaler.transform(x_train[test])
            target_train_ord = class_ordinal(y_train[train])
            target_test_ord = class_ordinal(y_train[test])
            mlp_acc, mlp_loss = mlp_model(data_train_norm, target_train_ord,\
             data_test_norm, target_test_ord)
            mlp_acc_tot.append(mlp_acc)
            mlp_loss_tot.append(mlp_loss)

    print("MLP KFold Epochs, Accuracy, loss: ", EPOCHS, mean(mlp_acc_tot), mean(mlp_loss_tot))

def main():
    """The main program that runs the necessary functions."""
    x_train, y_train = feature_engineering()
    kfold_validation(x_train, y_train)
    plt.figure(figsize=(12, 5))
    plt.show()

if __name__ == "__main__":
    main()
