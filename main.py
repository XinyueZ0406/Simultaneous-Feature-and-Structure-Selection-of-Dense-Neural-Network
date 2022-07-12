import glmnet_python
from glmnet import glmnet
import scipy, importlib, pprint, matplotlib.pyplot as plt, warnings
from glmnetPlot import glmnetPlot
from glmnetPrint import glmnetPrint
from glmnetCoef import glmnetCoef
from glmnetPredict import glmnetPredict
from cvglmnet import cvglmnet
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, concatenate, Activation
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import math
import tensorflow as tf
import tensorflow
from sklearn.model_selection import StratifiedKFold
import itertools
from sklearn.datasets import load_iris

# define early stopping criteria
class ES(tensorflow.keras.callbacks.Callback):
    def __init__(self, delta=0, previous=1000, patience=30):
        super(tensorflow.keras.callbacks.Callback, self).__init__()
        self.delta = delta
        self.previous = previous
        self.patience = patience

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_loss")
        if current - self.previous > 0:
            self.previous = current
            self.wait += 1
        else:
            if np.abs(current - self.previous) < self.delta:
                self.previous = current
                self.wait += 1

            else:
                self.previous = current
                self.wait = 0

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True

            # print(self.wait)


def relu_derivative(z):
    return np.greater(z, 0).astype(int)



def choose_alpha(mse_p, threshold=1, nfold, alphas, mode):
    p_m = mse_p.mean(axis=1)
    p_std = mse_p.std(axis=1)
    if mode == "1se":
        max_mse = p_m.min() + threshold * (p_std[np.where(p_m == p_m.min())[0][0]] / np.sqrt(nfold))
        alpha_c = alphas[np.where(p_m <= max_mse)].max()
    if mode == "min":
        alpha_c = alphas[np.where(p_m == p_m.min())].max()
    return (alpha_c)

def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.75
    epochs_drop = 20.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def DNN(h, type, X_train, Y_train, X_valid, Y_valid, epochs, batch_size):
    # h is the number of hidden layers in DNN
    # nc is the number of output nodes
    reduce_lr = tensorflow.keras.callbacks.LearningRateScheduler(step_decay)
    es = ES(delta=0.00001)
    inp_o = Input(shape=(X_train.shape[1],))
    inp = inp_o
    for i in range(1, (h + 1)):
        out = Dense(1, activation='relu', kernel_initializer='normal')(inp)
        inp = concatenate([inp, out])
    if type == "regression":
        nc = 1
        out_l = Dense(nc, kernel_initializer='normal')(inp)
        model = Model(inp_o, out_l)
        model.compile(loss='mse', optimizer=tensorflow.keras.optimizers.SGD(lr=0.01), metrics=['mse'])

    if type == "binary":
        nc = 1
        out_l = Dense(nc, kernel_initializer='normal', activation='sigmoid')(inp)
        model = Model(inp_o, out_l)
        model.compile(loss='binary_crossentropy', optimizer=tensorflow.keras.optimizers.SGD(lr=0.01),
                      metrics=['accuracy'])
    if type == "three_label":
        nc = Y_train.shape[1]
        out_l = Dense(nc, kernel_initializer='normal', activation='softmax')(inp)
        model = Model(inp_o, out_l)
        model.compile(loss='categorical_crossentropy', optimizer=tensorflow.keras.optimizers.SGD(lr=0.01),
                      metrics=['accuracy'])


    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_valid, Y_valid), verbose=2,
              callbacks=[reduce_lr, es])

    return (model)

def backward_lasso_selection(initial_lrate_r, drop_r, h, nfold, batch_size, model, X_train, Y_train, X_valid, Y_valid,
                             mode, type):
    # load DNN model as model
    # initial_lrate_r: define initial learning rate
    # drop_r: drop rate of learning rate
    # h: the number of hidden layers in DNN
    # mode: 1se or min when choosing alpha
    n_patience = 0
    epochs_drop_r = 1.0
    o_p = h + X_train.shape[1]
    te_previous = 10000
    nc = Y_train.shape[1]
    n_threshold = 200
    if type == "binary":
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    if type == "three_label":
        bce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    for nn in range(2000):
        # get weight matrix from DNN
        w_m = pd.DataFrame()
        bias = pd.DataFrame()
        k = X_train.shape[1]
        for lay in model.layers:
            if 'dense' in lay.name:
                p = pd.DataFrame(model.get_layer(name=lay.name).get_weights()[0])
                if p.shape[1] < nc:
                    p = pd.DataFrame(model.get_layer(name=lay.name).get_weights()[0], columns=[k])
                    w_m = pd.concat([w_m, p], axis=1)
                    pp = pd.DataFrame(model.get_layer(name=lay.name).get_weights()[1], columns=[k])
                    bias = pd.concat([bias, pp], axis=1)
                    k = k + 1
                else:
                    p = pd.DataFrame(model.get_layer(name=lay.name).get_weights()[0], columns=[k, k + 1, k + 2])
                    w_m = pd.concat([w_m, p], axis=1)
                    pp = pd.DataFrame(model.get_layer(name=lay.name).get_weights()[1].reshape(1, -1),
                                      columns=[k, k + 1, k + 2])
                    bias = pd.concat([bias, pp], axis=1)
                    k = k + 1


        # get ouput matrix
        get_all_layer_outputs = K.function([model.layers[0].input], [l.output for l in model.layers[1:]])
        layer_output = get_all_layer_outputs([X_train])
        output_m = pd.DataFrame(X_train)
        k = X_train.shape[1]
        for i in range(h + 1):
            lo = pd.DataFrame(layer_output[2 * i])
            if lo.shape[1] < nc:
                layerOutput = pd.DataFrame(layer_output[2 * i], columns=[k])
                output_m = pd.concat([output_m, layerOutput], axis=1)
                k += 1
            else:
                layerOutput = pd.DataFrame(layer_output[2 * i], columns=[k, k + 1, k + 2])
                output_m = pd.concat([output_m, layerOutput], axis=1)
                k += 1
        output_m.columns = np.arange(0, X_train.shape[1] + h + nc)

        if type == "regression":
            if mode == "1se":
                lp = "lambda_1se"
            if mode == "min":
                lp = "lambda_min"

            # output node
            cvfit = cvglmnet(x=np.array(output_m.iloc[:, 0:o_p]).copy(), y=Y_train.copy(), ptype='mse', nfolds=nfold,
                             family='gaussian', alpha=1, standardize=False)
            fit = glmnet(x=np.array(output_m.iloc[:, 0:o_p]).copy(), y=Y_train.copy(), family='gaussian', alpha=1,
                         standardize=False)
            coef = glmnetCoef(fit, s=cvfit[lp], exact=False)
            w_m.loc[np.asarray(np.where(np.abs(coef[1:o_p + 1]) == 0))[0], o_p] = 0
            mw = model.get_weights()[(2 * h):(2 * (h + 1))]
            mw[0] = np.array(w_m.iloc[:, h].dropna()).reshape(-1, 1)
            model.layers[(2 * h) + 1].set_weights(mw)
            y_pred_t = model.predict(X_train)
            get_all_layer_outputs = K.function([model.layers[0].input], [l.output for l in model.layers[1:]])
            layer_output = get_all_layer_outputs([X_train])
            output_m[o_p] = layer_output[2 * h]
            # anther hidden nodes
            for nh in reversed(range(X_train.shape[1], X_train.shape[1] + h)):
                features = range(nh, (X_train.shape[1] + h))
                v = np.zeros((len(features), len(features), X_train.shape[0]))
                for k in features:
                    d_k = relu_derivative(
                        np.dot(output_m.loc[:, 0:(k - 1)], [x for x in w_m[k] if str(x) != 'nan']) + np.repeat(bias[k],
                                                                                                               X_train.shape[
                                                                                                                   0]))
                    for j in range((k + 1), (X_train.shape[1] + h + 1)):
                        v[k - nh][j - nh - 1] = np.multiply(d_k, w_m.loc[k, j])
                u = pd.DataFrame(np.ones(X_train.shape[0]), columns=[X_train.shape[1] + h])
                for hh in reversed(range(nh, X_train.shape[1] + h)):
                    u[hh] = np.zeros(X_train.shape[0])
                    for i in range(hh, X_train.shape[1] + h):
                        u[hh] += np.multiply(u[i + 1], v[hh - nh][i - nh])
                az = (np.dot(output_m.loc[:, 0:(nh - 1)], [x for x in w_m[nh] if str(x) != 'nan']) + np.repeat(bias[nh],
                                                                                                               X_train.shape[
                                                                                                                   0]))
                diff_y = (Y_train.ravel() - list(y_pred_t.ravel())).reshape(-1, 1)
                z = [j / k if k else np.array([0]) for j, k in zip(diff_y, u[nh].values.reshape(-1, 1))]
                lasso_y = np.add(z, az.values.reshape(-1, 1))
                wts = u[nh] ** 2
                n_rows, n_cols = np.shape(output_m.iloc[:, 0:nh])
                if np.sum(u[nh] == 0) == X_train.shape[0]:
                    w_m.loc[0:n_cols - 1, nh] = 0
                else:
                    lasso_x = output_m.iloc[:, 0:nh]
                    cvfit2 = cvglmnet(x=np.array(lasso_x).copy(), y=lasso_y.copy(), weights=np.array(wts), ptype='mse',
                                      nfolds=nfold, family='gaussian', alpha=1, standardize=False)
                    fit2 = glmnet(x=np.array(lasso_x).copy(), y=lasso_y.copy(), weights=np.array(wts),
                                  family='gaussian', alpha=1, standardize=False)
                    coef2 = glmnetCoef(fit2, s=cvfit2[lp], exact=False)
                    w_m.loc[np.asarray(np.where(np.abs(coef2[1:n_cols + 1]) == 0))[0], nh] = 0

                mw = model.get_weights()[(2 * (nh - X_train.shape[1])):(2 * ((nh - X_train.shape[1]) + 1))]
                mw[0] = np.array(w_m.iloc[:, (nh - X_train.shape[1])].dropna()).reshape(-1, 1)
                model.layers[(2 * (nh - X_train.shape[1])) + 1].set_weights(mw)
                y_pred_t = model.predict(X_train)
                get_all_layer_outputs = K.function([model.layers[0].input], [l.output for l in model.layers[1:]])
                layer_output = get_all_layer_outputs([X_train])
                for ii in range(nh, X_train.shape[1] + h + 1):
                    output_m[ii] = layer_output[2 * (ii - X_train.shape[1])]
        else:
            kf = StratifiedKFold(n_splits=nfold)
            loss_cv = pd.DataFrame()
            for train_index, valid_index in kf.split(np.array(output_m.iloc[:, 0:o_p]),
                                                     np.float64(Y_train.reshape(-1, 1))):
                X_train_n, X_valid_n = np.array(output_m.iloc[:, 0:o_p])[train_index], \
                                       np.array(output_m.iloc[:, 0:o_p])[
                                           valid_index]
                Y_train_n, Y_valid_n = np.float64(Y_train.reshape(-1, 1))[train_index], \
                                       np.float64(Y_train.reshape(-1, 1))[
                                           valid_index]
                fit = glmnet(x=X_train_n.copy(), y=Y_train_n.copy(),
                             family='binomial', alpha=1, standardize=False)
                p = glmnetPredict(fit, newx=X_valid_n.copy(), ptype='response', s=scipy.array(np.logspace(-6, 2, 200)))
                loss_m = []
                for i in range(p.shape[1]):
                    loss_m.append(bce(Y_valid_n, p[:, i].reshape(-1, 1)).numpy())
                loss_cv = pd.concat([loss_cv, pd.DataFrame(loss_m).T])
            s = choose_alpha(loss_cv.T, threshold=1, nfold=nfold, alphas=np.logspace(-6, 2, 200), mode=mode)
            fit = glmnet(x=np.array(output_m.iloc[:, 0:o_p]).copy(), y=np.float64(Y_train.reshape(-1, 1)).copy(),
                         family='binomial', alpha=1, standardize=False)
            coef = glmnetCoef(fit, s=scipy.array([s]), exact=False)
            if type == "three_label":
                w_m.loc[np.asarray(np.where(np.abs(coef[0][1:o_p + 1]) == 0))[0], o_p] = 0
                w_m.loc[np.asarray(np.where(np.abs(coef[1][1:o_p + 1]) == 0))[0], o_p + 1] = 0
                w_m.loc[np.asarray(np.where(np.abs(coef[2][1:o_p + 1]) == 0))[0], o_p + 2] = 0
            if type == "binary":
                w_m.loc[np.asarray(np.where(np.abs(coef[1:o_p + 1]) == 0))[0], o_p] = 0
            # print(w_m)
            mw = model.get_weights()[(2 * h):(2 * (h + 1))]
            if type == "three_label":
                mw[0] = np.array(w_m.iloc[:, h:(h + nc)].dropna())
            if type == "binary":
                mw[0] = np.array(w_m.iloc[:, h].dropna()).reshape(-1, 1)
            model.layers[(2 * h) + 1].set_weights(mw)
            # print(model.get_weights())
            get_all_layer_outputs = K.function([model.layers[0].input], [l.output for l in model.layers[1:]])
            layer_output = get_all_layer_outputs([X_train])
            if type == "three_label":
                output_m.loc[:, o_p:(o_p + nc)] = layer_output[2 * h]
            if type == "binary":
                output_m[o_p] = layer_output[2 * h]
            # print(output_m)

            # anther hidden nodes
            if type == "three_label":
                for nh in reversed(range(X_train.shape[1], X_train.shape[1] + h)):
                    features = range(nh, (X_train.shape[1] + h))
                    w_m1 = w_m.loc[:, X_train.shape[1]:(X_train.shape[1] + h)]
                    bias1 = bias.loc[:, X_train.shape[1]:(X_train.shape[1] + h)]
                    v = np.zeros((len(features), len(features), X_train.shape[0]))
                    for k in features:
                        d_k = relu_derivative(
                            np.dot(output_m.loc[:, 0:(k - 1)], [x for x in w_m1[k] if str(x) != 'nan']) + np.repeat(
                                bias1[k],
                                X_train.shape[
                                    0]))
                        for j in range((k + 1), (X_train.shape[1] + h + 1)):
                            v[k - nh][j - nh - 1] = np.multiply(d_k, w_m1.loc[k, j])
                    u1 = pd.DataFrame(np.ones(X_train.shape[0]), columns=[X_train.shape[1] + h])
                    for hh in reversed(range(nh, X_train.shape[1] + h)):
                        u1[hh] = np.zeros(X_train.shape[0])
                        for i in range(hh, X_train.shape[1] + h):
                            u1[hh] += np.multiply(u1[i + 1], v[hh - nh][i - nh])
                    ic = list(
                        itertools.chain(range(X_train.shape[1], X_train.shape[1] + h), [X_train.shape[1] + h + 1]))
                    w_m2 = w_m.loc[:, ic]
                    w_m2.columns = np.arange(X_train.shape[1], X_train.shape[1] + h + 1)
                    bias2 = bias.loc[:, ic]
                    bias2.columns = np.arange(X_train.shape[1], X_train.shape[1] + h + 1)
                    v = np.zeros((len(features), len(features), X_train.shape[0]))
                    for k in features:
                        d_k = relu_derivative(
                            np.dot(output_m.loc[:, 0:(k - 1)], [x for x in w_m2[k] if str(x) != 'nan']) + np.repeat(
                                bias2[k],
                                X_train.shape[
                                    0]))
                        for j in range((k + 1), (X_train.shape[1] + h + 1)):
                            v[k - nh][j - nh - 1] = np.multiply(d_k, w_m2.loc[k, j])
                    u2 = pd.DataFrame(np.ones(X_train.shape[0]), columns=[X_train.shape[1] + h])
                    for hh in reversed(range(nh, X_train.shape[1] + h)):
                        u2[hh] = np.zeros(X_train.shape[0])
                        for i in range(hh, X_train.shape[1] + h):
                            u2[hh] += np.multiply(u2[i + 1], v[hh - nh][i - nh])

                    ic = list(
                        itertools.chain(range(X_train.shape[1], X_train.shape[1] + h), [X_train.shape[1] + h + 2]))
                    w_m3 = w_m.loc[:, ic]
                    w_m3.columns = np.arange(X_train.shape[1], X_train.shape[1] + h + 1)
                    bias3 = bias.loc[:, ic]
                    bias3.columns = np.arange(X_train.shape[1], X_train.shape[1] + h + 1)
                    v = np.zeros((len(features), len(features), X_train.shape[0]))
                    for k in features:
                        d_k = relu_derivative(
                            np.dot(output_m.loc[:, 0:(k - 1)], [x for x in w_m3[k] if str(x) != 'nan']) + np.repeat(
                                bias3[k],
                                X_train.shape[
                                    0]))
                        for j in range((k + 1), (X_train.shape[1] + h + 1)):
                            v[k - nh][j - nh - 1] = np.multiply(d_k, w_m3.loc[k, j])
                    u3 = pd.DataFrame(np.ones(X_train.shape[0]), columns=[X_train.shape[1] + h])
                    for hh in reversed(range(nh, X_train.shape[1] + h)):
                        u3[hh] = np.zeros(X_train.shape[0])
                        for i in range(hh, X_train.shape[1] + h):
                            u3[hh] += np.multiply(u3[i + 1], v[hh - nh][i - nh])

                    az = (
                    np.dot(output_m.loc[:, 0:(nh - 1)], [x for x in w_m[nh] if str(x) != 'nan']) + np.repeat(bias[nh],
                                                                                                             X_train.shape[
                                                                                                                 0]))

                    yhat1 = (
                        np.dot(output_m.loc[:, 0:o_p - 1], [x for x in w_m[o_p]]) + np.repeat(bias[o_p],
                                                                                              X_train.shape[0]))
                    offset1 = yhat1.values.reshape(-1, 1) - np.multiply(np.asarray(az), u1[nh]).values.reshape(-1, 1)

                    yhat2 = (
                        np.dot(output_m.loc[:, 0:o_p - 1], [x for x in w_m[o_p + 1]]) + np.repeat(bias[o_p + 1],
                                                                                                  X_train.shape[0]))
                    offset2 = yhat2.values.reshape(-1, 1) - np.multiply(np.asarray(az), u2[nh]).values.reshape(-1, 1)

                    yhat3 = (
                        np.dot(output_m.loc[:, 0:o_p - 1], [x for x in w_m[o_p + 2]]) + np.repeat(bias[o_p + 2],
                                                                                                  X_train.shape[0]))
                    offset3 = yhat3.values.reshape(-1, 1) - np.multiply(np.asarray(az), u3[nh]).values.reshape(-1, 1)

                    n_rows, n_cols = np.shape(output_m.iloc[:, 0:nh])
                    if (np.sum(u1[nh] == 0) + np.sum(u2[nh] == 0) + np.sum(u3[nh] == 0)) == X_train.shape[0] * 3:
                        w_m.loc[0:n_cols - 1, nh] = 0

                    else:
                        X_intercept = np.append(output_m.iloc[:, 0:nh], np.ones([n_rows, 1]), axis=1)
                        lasso_x1 = np.dot(np.diag(list(u1[nh])), X_intercept)
                        lasso_x2 = np.dot(np.diag(list(u2[nh])), X_intercept)
                        lasso_x3 = np.dot(np.diag(list(u3[nh])), X_intercept)
                        ya_0 = np.where(Y_train[:, 0] == 0)
                        y_hat2 = np.multiply(u2[nh].values.reshape(-1, 1),
                                             (np.dot(output_m.loc[:, 0:(nh - 1)],
                                                     [x for x in w_m[nh] if str(x) != 'nan'])
                                              + np.repeat(bias[nh], X_train.shape[0])).values.reshape(-1, 1))
                        y_hat3 = np.multiply(u3[nh].values.reshape(-1, 1),
                                             (np.dot(output_m.loc[:, 0:(nh - 1)],
                                                     [x for x in w_m[nh] if str(x) != 'nan'])
                                              + np.repeat(bias[nh], X_train.shape[0])).values.reshape(-1, 1))
                        vv = np.log(np.exp(y_hat2 + offset2) + np.exp(y_hat3 + offset3))
                        lasso_y = np.vstack([Y_train[:, 0].reshape(-1, 1), Y_train[ya_0[0], 1].reshape(-1, 1)])
                        lasso_x = np.vstack([lasso_x1, lasso_x2[ya_0[0], :] - lasso_x3[ya_0[0], :]])
                        lasso_os = np.vstack([offset1 - vv, offset2[ya_0[0]] - offset3[ya_0[0]]])
                        kf = StratifiedKFold(n_splits=5)
                        loss_cv = pd.DataFrame()
                        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
                        for train_index, valid_index in kf.split(np.array(lasso_x), np.float64(lasso_y)):
                            X_train_n, X_valid_n = np.array(lasso_x)[train_index], np.array(lasso_x)[valid_index]
                            Y_train_n, Y_valid_n = np.float64(lasso_y)[train_index], np.float64(lasso_y)[valid_index]
                            offset_t, offset_v = lasso_os[train_index], lasso_os[valid_index]
                            fit = glmnet(x=X_train_n.copy(), y=Y_train_n.copy(), family='binomial', alpha=1,
                                         standardize=False, intr=0, offset=offset_t,
                                         penalty_factor=np.append(np.ones([nh, 1]), 0),
                                         lambdau=np.array(np.logspace(-6, 2, 10)))
                            p = glmnetPredict(fit, newx=X_valid_n.copy(), ptype='response', offset=offset_v,
                                              s=np.array(np.logspace(-6, 2, 200)))
                            loss_m = []
                            for i in range(p.shape[1]):
                                loss_m.append(bce(Y_valid_n, p[:, i].reshape(-1, 1)).numpy())
                            loss_cv = pd.concat([loss_cv, pd.DataFrame(loss_m).T])
                        s = choose_alpha(loss_cv.T, threshold=1, nfold=nfold, alphas=np.logspace(-6, 2, 200), mode=mode)
                        fit2 = glmnet(x=np.array(lasso_x).copy(), y=np.float64(lasso_y).copy(),
                                      family='binomial', alpha=1, standardize=False, intr=0, offset=lasso_os,
                                      penalty_factor=np.append(np.ones([nh, 1]), 0),
                                      lambdau=np.array(np.logspace(-6, 2, 10)))
                        coef2 = glmnetCoef(fit2, s=np.array([s]), exact=False)
                        w_m.loc[np.asarray(np.where(np.abs(coef2[1:n_cols + 1]) == 0))[0], nh] = 0
                    mw = model.get_weights()[(2 * (nh - X_train.shape[1])):(2 * ((nh - X_train.shape[1]) + 1))]
                    mw[0] = np.array(w_m.iloc[:, (nh - X_train.shape[1])].dropna()).reshape(-1, 1)
                    model.layers[(2 * (nh - X_train.shape[1])) + 1].set_weights(mw)
                    get_all_layer_outputs = K.function([model.layers[0].input], [l.output for l in model.layers[1:]])
                    layer_output = get_all_layer_outputs([X_train])
                    for ii in range(nh, X_train.shape[1] + h):
                        output_m[ii] = layer_output[2 * (ii - X_train.shape[1])]
                        # output_m.loc[:, o_p:(o_p + nc)] = layer_output[2 * h]

            if type == "binary":
                for nh in reversed(range(X_train.shape[1], X_train.shape[1] + h)):
                    features = range(nh, (X_train.shape[1] + h))
                    w_m1 = w_m.loc[:, X_train.shape[1]:(X_train.shape[1] + h)]
                    bias1 = bias.loc[:, X_train.shape[1]:(X_train.shape[1] + h)]
                    v = np.zeros((len(features), len(features), X_train.shape[0]))
                    for k in features:
                        d_k = relu_derivative(
                            np.dot(output_m.loc[:, 0:(k - 1)], [x for x in w_m[k] if str(x) != 'nan']) + np.repeat(
                                bias[k],
                                X_train.shape[
                                    0]))
                        for j in range((k + 1), (X_train.shape[1] + h + 1)):
                            v[k - nh][j - nh - 1] = np.multiply(d_k, w_m.loc[k, j])
                    u = pd.DataFrame(np.ones(X_train.shape[0]), columns=[X_train.shape[1] + h])
                    for hh in reversed(range(nh, X_train.shape[1] + h)):
                        u[hh] = np.zeros(X_train.shape[0])
                        for i in range(hh, X_train.shape[1] + h):
                            u[hh] += np.multiply(u[i + 1], v[hh - nh][i - nh])

                    az = (
                    np.dot(output_m.loc[:, 0:(nh - 1)], [x for x in w_m[nh] if str(x) != 'nan']) + np.repeat(bias[nh],
                                                                                                             X_train.shape[
                                                                                                                 0]))

                    yhat = (
                        np.dot(output_m.loc[:, 0:o_p - 1], [x for x in w_m[o_p]]) + np.repeat(bias[o_p],
                                                                                              X_train.shape[0]))

                    offset = yhat.values.reshape(-1, 1) - np.multiply(np.asarray(az), u[nh]).values.reshape(-1, 1)
                    n_rows, n_cols = np.shape(output_m.iloc[:, 0:nh])

                    if np.sum(u[nh] == 0) == X_train.shape[0]:
                        w_m.loc[0:n_cols - 1, nh] = 0

                    else:
                        X_intercept = np.append(output_m.iloc[:, 0:nh], np.ones([n_rows, 1]), axis=1)
                        lasso_x = np.dot(np.diag(list(u[nh])), X_intercept)
                        kf = StratifiedKFold(n_splits=nfold)
                        loss_cv = pd.DataFrame()
                        for train_index, valid_index in kf.split(np.array(lasso_x), np.float64(Y_train.reshape(-1, 1))):
                            X_train_n, X_valid_n = np.array(lasso_x)[train_index], np.array(lasso_x)[valid_index]
                            Y_train_n, Y_valid_n = np.float64(Y_train.reshape(-1, 1))[train_index], \
                                                   np.float64(Y_train.reshape(-1, 1))[valid_index]
                            offset_t, offset_v = offset[train_index], offset[valid_index]
                            fit = glmnet(x=X_train_n.copy(), y=Y_train_n.copy(), family='binomial', alpha=1,
                                         standardize=False, intr=0, offset=offset_t,
                                         penalty_factor=np.append(np.ones([nh, 1]), 0))
                            p = glmnetPredict(fit, newx=X_valid_n.copy(), ptype='response', offset=offset_v,
                                              s=scipy.array(np.logspace(-6, 2, 200)))
                            loss_m = []
                            for i in range(p.shape[1]):
                                loss_m.append(bce(Y_valid_n, p[:, i].reshape(-1, 1)).numpy())
                            loss_cv = pd.concat([loss_cv, pd.DataFrame(loss_m).T])
                        s = choose_alpha(loss_cv.T, threshold=1, nfold=nfold, alphas=np.logspace(-6, 2, 200), mode=mode)
                        fit2 = glmnet(x=np.array(lasso_x).copy(), y=np.float64(Y_train.reshape(-1, 1)).copy(),
                                      family='binomial', alpha=1, standardize=False, intr=0, offset=offset,
                                      penalty_factor=np.append(np.ones([nh, 1]), 0))
                        coef2 = glmnetCoef(fit2, s=scipy.array([s]), exact=False)
                        w_m.loc[np.asarray(np.where(np.abs(coef2[1:n_cols + 1]) == 0))[0], nh] = 0

                    # print(w_m)
                    mw = model.get_weights()[(2 * (nh - X_train.shape[1])):(2 * ((nh - X_train.shape[1]) + 1))]
                    mw[0] = np.array(w_m.iloc[:, (nh - X_train.shape[1])].dropna()).reshape(-1, 1)
                    model.layers[(2 * (nh - X_train.shape[1])) + 1].set_weights(mw)
                    # print(model.get_weights())
                    get_all_layer_outputs = K.function([model.layers[0].input], [l.output for l in model.layers[1:]])
                    layer_output = get_all_layer_outputs([X_train])
                    for ii in range(nh, X_train.shape[1] + h + 1):
                        output_m[ii] = layer_output[2 * (ii - X_train.shape[1])]
                        # print(output_m)

        # result
        va1 = model.evaluate(X_valid, Y_valid, verbose=0)[0]

        if np.abs(va1 - te_previous) < 0.0001 and np.std(history.history['val_loss']) < 0.00001:
            n_patience += 1
            te_previous = va1
        else:
            te_previous = va1
            n_patience = 0

        if n_patience > 8:

            break
        # extra one epoch
        lrate_r = initial_lrate_r * math.pow(drop_r, math.floor((1 + nn) / epochs_drop_r))
        K.set_value(model.optimizer.lr, lrate_r)
        history = model.fit(X_train, Y_train, epochs=10, batch_size=batch_size, validation_data=(X_valid, Y_valid),
                            verbose=2)
    return (model)


def get_reduced_model(model, X_train, Y_train, type):
    w_m = pd.DataFrame()
    bias = pd.DataFrame()
    k = X_train.shape[1]
    nc = Y_train.shape[1]
    for lay in model.layers:
        if 'dense' in lay.name:
            p = pd.DataFrame(model.get_layer(name=lay.name).get_weights()[0])
            if p.shape[1] < nc:
                p = pd.DataFrame(model.get_layer(name=lay.name).get_weights()[0], columns=[k])
                w_m = pd.concat([w_m, p], axis=1)
                pp = pd.DataFrame(model.get_layer(name=lay.name).get_weights()[1], columns=[k])
                bias = pd.concat([bias, pp], axis=1)
                k = k + 1
            else:
                p = pd.DataFrame(model.get_layer(name=lay.name).get_weights()[0], columns=[k, k + 1, k + 2])
                w_m = pd.concat([w_m, p], axis=1)
                pp = pd.DataFrame(model.get_layer(name=lay.name).get_weights()[1].reshape(1, -1),
                                  columns=[k, k + 1, k + 2])
                bias = pd.concat([bias, pp], axis=1)
                k = k + 1

    w2 = pd.DataFrame()
    w2 = w2.append(w_m)
    # drop columns and rows with all nan value
    for q in range(10000):
        w2 = w2.loc[w2.index[~w2.isin(['NaN', 0]).all(axis=1)], w2.columns[~w2.isin(['NaN', 0]).all()]]
        w2 = w2.replace(0, np.nan)
        l_0 = [value for value in w2.columns[:-1] if value not in w2.iloc[:, -1].index]
        w2 = w2.drop(l_0, axis=1)
        l_1 = [value for value in w2.iloc[:, -1].index if value not in w2.columns]
        ll_1 = [value for value in l_1 if value not in range(0, X_train.shape[1])]
        if len(ll_1) > 0:
            for lll in ll_1:
                lll_l = w2.loc[lll, :].index[~w2.loc[lll, :].isin(['NaN', 0])]
                for llll in list(lll_l):
                    bias.loc[0, llll] += w2.loc[lll, llll] * bias.loc[0, lll]
            w2 = w2.drop(ll_1, axis=0)
        if len(l_0) == 0 and len(ll_1) == 0:
            break
    #create reduced neural network
    inp = Input(shape=(X_train.shape[1],))
    for i in range(X_train.shape[1]):
        inp_i = "inp" + str(i)
        exec(inp_i + "= Lambda(lambda x: x[:, {0}:({0}+1)])(inp)".format(i))

    out_lll = []
    for i in w2.columns:
        out = "out" + str(i)
        l = list(w2[i][w2[i].notnull()].index)

        if i < (h + X_train.shape[1]):
            if set(l) <= set(range(0, X_train.shape[1])):
                if len(l) > 1:
                    inp_multi = ",".join(["inp" + str(j) for j in l])
                    exec("inp_c = concatenate([" + inp_multi + "])")
                    exec(out + "=Dense(1, activation='relu',name='dd{}')(inp_c)".format(i))
                else:
                    inp_one = "inp" + str(l[0])
                    exec(out + "=Dense(1, activation='relu',name='dd{}')(".format(i) + inp_one + ")")
            else:
                l_inp = [value for value in l if value in range(0, X_train.shape[1])]
                l_hn = [value for value in l if value not in l_inp]
                if len(l_inp) > 0:
                    inp_multi = ",".join(["inp" + str(j) for j in l_inp])
                    inp_out = ",".join(["out" + str(j) for j in l_hn])
                    exec("inp_c = concatenate([" + inp_multi + ", " + inp_out + "])")
                    exec(out + "=Dense(1, activation='relu',name='dd{}')(inp_c)".format(i))
                else:
                    if len(l_hn) > 1:
                        inp_out = ",".join(["out" + str(j) for j in l_hn])
                        exec("inp_c = concatenate([" + inp_out + "])")
                        exec(out + "=Dense(1, activation='relu',name='dd{}')(inp_c)".format(i))
                    else:
                        inp_out_one = "out" + str(l_hn[0])
                        exec(out + "=Dense(1, activation='relu',name='dd{}')(".format(i) + inp_out_one + ")")

        else:
            out_lll.append(i)
            l_inp = [value for value in l if value in range(0, X_train.shape[1])]
            l_hn = [value for value in l if value not in l_inp]
            if len(l_inp) > 0:
                inp_multi = ",".join(["inp" + str(j) for j in l_inp])
                inp_out = ",".join(["out" + str(j) for j in l_hn])
                exec("inp_c = concatenate([" + inp_multi + ", " + inp_out + "])")
                exec(out + "=Dense(1,name='dd{}')(inp_c)".format(i))
            else:

                if len(l_hn) > 1:
                    inp_out = ",".join(["out" + str(j) for j in l_hn])
                    exec("inp_c = concatenate([" + inp_out + "])")
                    exec(out + "=Dense(1, name='dd{}')(inp_c)".format(i))
                else:
                    inp_out_one = "out" + str(l_hn[0])
                    exec(out + "=Dense(1, name='dd{}')(".format(i) + inp_out_one + ")")
    if type == "regression":
        out = ",".join(["out" + str(j) for j in out_lll])
        exec("lasso_model = Model(inp," + out +")")
        lasso_model.compile(loss='mse', optimizer=tensorflow.keras.optimizers.SGD(lr=0.01), metrics=['mse'])


    if type == "binary":
        out = ",".join(["out" + str(j) for j in out_lll])
        exec("outt = Activation('sigmoid')(" + out + ")")
        lasso_model = Model(inp, outt)
        lasso_model.compile(loss='binary_crossentropy', optimizer=tensorflow.keras.optimizers.SGD(lr=0.01),
                        metrics=['accuracy'])
    if type == "three_label":
        if len(out_lll) > 1:
            out_final = ",".join(["out" + str(j) for j in out_lll])
            exec("out = concatenate([" + out_final + "])")
            outt = Activation('softmax')(out)
        else:
            out = ",".join(["out" + str(j) for j in out_lll])
            exec("outt = Activation('softmax')(" + out + ")")
        red_model = Model(inp, outt)

        red_model.compile(loss='categorical_crossentropy', optimizer=tensorflow.keras.optimizers.SGD(lr=0.01),
                            metrics=['accuracy'])


    for lay in red_model.layers:
        if 'dd' in lay.name:
            c = int(''.join(filter(lambda x: x.isdigit(), lay.name)))
            l = []
            x = np.array(w2[c].dropna()).reshape(-1, 1)
            if np.all((x == 0)):
                y = np.array([0.0])
            else:
                y = np.array(bias[c])
            l.append(x)
            l.append(y)
            red_model.get_layer(name='dd' + str(c)).set_weights(l)

    return (red_model)

def hard_threshold(diff_value, lasso_model, X_train, Y_train, X_valid, Y_valid, initial_lrate_r = 0.01, drop_r = 0.75,
                   epochs_drop_r = 30.0, batch_size):
    obj = lasso_model.evaluate(X_train, Y_train, verbose=0)[0]
    n_threshold = 200
    test_threshold = 100
    result_all = []
    te_previous = 10000
    n_patience = 0
    for nnn in range(10000):
        lrate_r = initial_lrate_r * math.pow(drop_r, math.floor((1 + nnn) / epochs_drop_r))
        K.set_value(lasso_model.optimizer.lr, lrate_r)

        w3 = pd.DataFrame()
        for lay in lasso_model.layers:
            if 'dd' in lay.name:
                c = int(''.join(filter(lambda x: x.isdigit(), lay.name)))
                p = pd.DataFrame(lasso_model.get_layer(name=lay.name).get_weights()[0], columns=[c])
                w3 = pd.concat([w3, p], axis=1)
        w3_rep = pd.DataFrame()
        w3_rep = w3_rep.append(w3)
        w3_rep[w3_rep == 0] = np.nan
        num_w3 = (np.abs(w3_rep) >= 0).sum().sum()
        # sort weight of weight matrix
        r = []
        c = []
        print(num_w3)
        for kk in range(0, num_w3):
            rr, cc = np.where(np.abs(w3_rep) == np.min(np.abs(w3_rep).min()))
            r.append(rr[0])
            c.append(cc[0])
            w3_rep.iloc[rr[0], cc[0]] = np.nan
        n_0 = 0
        # set weight to zero in order
        for k in range(0, num_w3):
            i = c[k]
            j = r[k]
            col_n = w3.columns[i]
            a = lasso_model.get_weights()[(2 * i):(2 * (i + 1))]
            a[0][j] = 0
            lasso_model.get_layer(name='dd' + str(col_n)).set_weights(a)
            obj_train = lasso_model.evaluate(X_train, Y_train, verbose=0)[0]
            diff = obj_train - obj
            if diff > diff_value:
                a[0][j] = w3.iloc[j, i]

                lasso_model.get_layer(name='dd' + str(col_n)).set_weights(a)
                n_0 = n_0 + 1
            obj = lasso_model.evaluate(X_train, Y_train, verbose=0)[0]

        # w4 after one by one drop
        va1 = lasso_model.evaluate(X_valid, Y_valid, verbose=0)[0]

        if n_0 != n_threshold:
            n_threshold = n_0
            n_patience = 0
            te_previous = va1

        else:
            if np.abs(va1 - te_previous) < 0.00001:
                n_patience += 1
                te_previous = va1

            else:
                te_previous = va1
                n_patience = 0

        if n_patience > 30:

            break
        # do one more epoch to adjust all weight
        lasso_model.fit(X_train, Y_train, epochs=1, batch_size=batch_size, validation_data=(X_valid, Y_valid),
                        verbose=2)
    return (lasso_model)


if __name__ == '__main__':
    iris = load_iris()
    X = iris['data']
    y = iris['target']
    names = iris['target_names']
    feature_names = iris['feature_names']
    Y = OneHotEncoder().fit_transform(y[:, np.newaxis]).toarray()
    #data prepocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    nc = Y.shape[1]
    h = 10
    print(nc)

    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, stratify=Y)
    result = []
    # fit DNN basic model
    model = DNN(h = 10, type = "three_label", X_train = X_train, Y_train = Y_train, X_valid = X_test, Y_valid = Y_test,
                epochs = 1000, batch_size = 16)
    tr1 = model.evaluate(X_train, Y_train, verbose=0)[0]
    tr2 = model.evaluate(X_train, Y_train, verbose=0)[1]
    te1 = model.evaluate(X_test, Y_test, verbose=0)[0]
    te2 = model.evaluate(X_test, Y_test, verbose=0)[1]
    result_basic = [tr1, tr2, te1, te2]
    print(result_basic)
    result.append(result_basic)

    # do backward lasso selection
    lasso_model = backward_lasso_selection(initial_lrate_r = 0.1, drop_r =0.86, h = 10, nfold =5,
                                           batch_size = 16, model = model, X_train = X_train, Y_train = Y_train,
                                           X_valid = X_test, Y_valid = Y_test, mode = "1se", type = "three_label")

    tr1 = lasso_model.evaluate(X_train, Y_train, verbose=0)[0]
    tr2 = lasso_model.evaluate(X_train, Y_train, verbose=0)[1]
    te1 = lasso_model.evaluate(X_test, Y_test, verbose=0)[0]
    te2 = lasso_model.evaluate(X_test, Y_test, verbose=0)[1]

    result_red = [tr1, tr2, te1, te2]

    print(result_red)
    result.append(result_red)

    # drop all zero-links from backward lasso selection step, and get the reduced model 1
    reduced_model = get_reduced_model(model = lasso_model, X_train = X_train, Y_train =Y_train, type = "three_label")

    # give cut-off values, here we can use cross validation to choose parameter
    diff_list = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0]
    for diff_value in diff_list:
        diff_model = hard_threshold(diff_value = diff_value, lasso_model = reduced_model, X_train = X_train, Y_train = Y_train,
                                    X_valid = X_test, Y_valid = Y_test, initial_lrate_r = 0.01, drop_r = 0.75,
                                    epochs_drop_r = 30.0, batch_size = 16)
        tr1 = diff_model.evaluate(X_train, Y_train, verbose=0)[0]
        tr2 = diff_model.evaluate(X_train, Y_train, verbose=0)[1]
        te1 = diff_model.evaluate(X_test, Y_test, verbose=0)[0]
        te2 = diff_model.evaluate(X_test, Y_test, verbose=0)[1]
        result_diff = [tr1, tr2, te1, te2]
        print(result_diff)
        result.append(result_diff)

    # drop all zero-links from hard threshold step, and get the final reduced model
    final_reduced_model = get_reduced_model(model=diff_model, X_train=X_train, Y_train=Y_train, type="three_label")





















