# import os.path

import pyriemann
import numpy as np
from mne.io import read_raw_edf
from pyriemann.utils.distance import distance
from pyriemann.utils.mean import mean_covariance
import math
import time
import csv
import pandas

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from torch import optim

import matplotlib.pyplot as plt
import LSLscripts.LSLacquire as LSLa

import eeg_io_pp
import eeg_io_pp_2
import Deep_Func

def get_clf():
    if clf_method == "Riemann":
        fgda = pyriemann.tangentspace.FGDA()
        mdm = pyriemann.classification.MDM()
        clf = Pipeline([('FGDA', fgda), ('MDM', mdm)])
        return clf, False
    elif clf_method == "Braindecode":
        model = Deep_Func.create_model(n_classes, freq*window_size, in_chans=num_channels)
        optimizer = optim.Adam(model.parameters())
        return model, optimizer
    elif clf_method == "LSTM":
        model = Deep_Func.create_model_lstm(n_classes, freq*window_size, in_chans=num_channels)
        optimizer = optim.Adam(model.parameters())
        return model, optimizer


def transform_fit(clf, opt, train_data, train_labels):
    np.set_printoptions(precision=3)
    fgda = pyriemann.tangentspace.FGDA()
    if clf_method == "Riemann":
        cov_train = pyriemann.estimation.Covariances().fit_transform(np.transpose(train_data, axes=[0, 2, 1]))
        # clf.fit_transform(cov_train, train_labels)    # if certainty is not needed
        cov_train = fgda.fit_transform(cov_train, train_labels)
        full_calc_mean_cov(cov_train, train_labels)
        return clf
    elif clf_method == "Braindecode":
        train_data = (train_data * 1e6).astype(np.float32)
        y = train_labels.astype(np.int64)
        X = np.transpose(train_data, [0, 2, 1])
        model = Deep_Func.fit_transform(clf, opt, X, y, input_time_length=int(freq*window_size), n_channels=num_channels,
                                        num_epochs=epochs)
        return model
    elif clf_method == "LSTM":
        train_data = (train_data * 1e6).astype(np.float32)
        y = train_labels.astype(np.int64)
        X = np.transpose(train_data, [0, 2, 1])
        model = Deep_Func.fit_transform(clf, opt, X, y, input_time_length=int(freq * window_size),
                                        n_channels=num_channels, num_epochs=epochs)
        return model


def predict(clf, val_data, labels):
    if clf_method == "Riemann":
        fgda = pyriemann.tangentspace.FGDA()
        cert = 1
        cov_val = pyriemann.estimation.Covariances().fit_transform(np.transpose(val_data, axes=[0, 2, 1]))
        # pred_val = clf.predict(cov_val)
        pred_val, cert = predict_Riemann(cov_val)
        # if np.isnan(cert[0]):
        #     print(val_data)
        return pred_val, cert
    elif clf_method == "Braindecode":
        val_data = (val_data * 1e6).astype(np.float32)
        X = np.transpose(val_data, [0, 2, 1])
        pred_val, cert = Deep_Func.predict(clf, X, labels, input_time_length=int(freq*window_size), n_channels=num_channels)
        return pred_val, cert
    elif clf_method == "LSTM":
        val_data = (val_data * 1e6).astype(np.float32)
        X = np.transpose(val_data, [0, 2, 1])
        pred_val, cert = Deep_Func.predict(clf, X, labels, input_time_length=int(freq * window_size),
                                       n_channels=num_channels)
        return pred_val, cert


def full_calc_mean_cov(cov_train_i, label_train_i, num_classes=4):

    global mean_cov_n
    mean_cov_n = np.zeros((num_classes, cov_train_i.shape[1], cov_train_i.shape[2]))
    mean_cov_i = mean_cov_n

    for l in range(num_classes):
        try:
            mean_cov_n[l] = mean_covariance(cov_train_i[label_train_i == l], metric='riemann',
                                            sample_weight=None)
            # print(mean_cov_n[l])
        except ValueError:
            mean_cov_n[l] = mean_cov_i[l]

    return mean_cov_n


def predict_Riemann(covtest):
    # print(covtest)
    dist = predict_distances_own(covtest)
    # print(dist)
    cert = (dist.mean(axis=1) - dist.min(axis=1))*4.0
    # if np.isnan(cert[0]):
    #     print(covtest)

    return dist.argmin(axis=1), cert


def predict_distances_own(covtest):
    covmeans = mean_cov_n
    Nc = len(covmeans)
    dist = [distance(covtest, covmeans[m], 'riemann') for m in range(Nc)]
    dist = np.concatenate(dist, axis=1)

    return dist


def get_data():

    global freq, dataset_dir, subject_name

    windowed = False

    if windowed:
        data_train_i = np.load(train_data_folder + train_file + '.npy')
        label_train_i = np.load(train_data_folder + train_file + '_labels.npy')
        print("Loading from datafile: ", train_data_folder + train_file)

        train_len = int(data_train_i.shape[0] * 0.667)
        data_train = data_train_i[:train_len]
        label_train = label_train_i[:train_len]
        data_val = data_train_i[train_len:]
        label_val = label_train_i[train_len:]

        # plt.hist(data_train.flatten(), bins=1000)
        # plt.show()

        for i in range(len(data_train)):
            # print(data_train[i, 16])
            data_train[i] = eeg_io_pp.butter_bandpass_filter(data_train[i], 7, 30, freq)
            data_train_ii = data_train[i]

        data_train = eeg_io_pp_2.norm_dataset(data_train)
    else:
        # data_dir = '/homes/df1215/Downloads/eeg_test_data/'
        # data_dir = '/data/EEG_Data/adaptive_eeg_test_data/'
        data_dir = 'D:/'
        # data = np.genfromtxt(data_dir + 'Daniel_0/df_FB_001_data.csv', delimiter=',')
        # labels = np.genfromtxt(data_dir + 'Daniel_0/df_FB_001_markers.csv', delimiter=',')
        # data = np.genfromtxt(data_dir + 'Fani_0/fd_FB_001_data.csv', delimiter=',')
        # labels = np.genfromtxt(data_dir + 'Fani_0/fd_FB_001_markers.csv', delimiter=',')
        data = np.genfromtxt(subject_name + '_EEG_data.csv', delimiter=',')
        labels = np.genfromtxt(subject_name + '_markers.csv', delimiter=',')

        data = data[1:len(data)]
        labels = labels[1:len(labels)]

        data_train_i, label_train_i = eeg_io_pp_2.label_data_lsl(data, labels, n_channels=num_channels)
        # print("shapes")
        # print(data_train_i.shape, label_train_i.shape)
        data_train, data_val, label_train, label_val = eeg_io_pp_2.process_data_2a(data_train_i, label_train_i, 250, num_channels=num_channels)
        # filter 5-13

    return data_train, label_train, data_val, label_val


def get_test_data():
    global data_test, num_channels

    dataset = "gtec"

    if dataset == "bci_comp":
        # data_test, label_test = eeg_io_pp.get_data_2a(dataset_dir + test_file, n_classes, remove_rest=False,
        #                                               training_data=False)
        raw = read_raw_edf(dataset_dir + test_file, preload=True, stim_channel='auto', verbose='WARNING')
        data_test = np.asarray(np.transpose(raw.get_data()[:num_channels]))
    elif dataset == "gtec":
        # data_test, label_test = eeg_io_pp.get_data_gtec(dataset_dir, file, n_classes)
        # data_dir = '/data/EEG_Data/adaptive_eeg_test_data/'
        num_channels = 32
        data_dir = 'D:/'
        # data = np.genfromtxt(data_dir + 'signal/' + file + '_01.csv', delimiter=';')
        data = np.genfromtxt(data_dir + 'Daniel_0/df_FB_001_data.csv', delimiter=',')
        # data = np.genfromtxt(data_dir + 'Fani_0/fd_FB_001_data.csv', delimiter=',')
        raw_data = np.zeros((len(data), num_channels))
        for i in range(1, len(data)):
            if not math.isnan(np.amax(data[i][1:num_channels + 1])):
                raw_data[i] = data[i][1:num_channels + 1]
        data_test = np.asarray(raw_data)
    return data_test


def init_globals(expInfo):
    global dataset, train_data_folder, train_file, model_file
    global dataset_dir, train_file, test_file
    global remove_rest_val, reuse_data, mult_data, noise_data, neg_data, da_mod
    global window_size, freq, num_channels, overlap, buffer_size
    global clf_method, n_classes, epochs
    global bci_iter

    bci_iter = 0

    clf_method = "Riemann"
    # clf_method = "Braindecode"
    # clf_method = "LSTM"

    n_classes = 2
    epochs = 10

    # dataset = "bci_comp"
    dataset = "gtec"

    running = True
    remove_rest_val = True
    reuse_data = False
    mult_data = False
    noise_data = False
    neg_data = False
    da_mod = 2

    subject_num = 7
    window_size = 0.5  # in seconds
    overlap = 2

    if dataset == "bci_comp":
        freq, num_channels = 250, 22
    elif dataset == "physionet":
        freq, num_channels = 160, 64
    elif dataset == "gtec":
        freq, num_channels = 250, 32

    buffer_size = int(freq * window_size)

    # File naming
    data_folder = '/homes/df1215/bci_test_venv/bin/'
    train_data_folder = data_folder + '4class_MI/data/final_data/'
    fb_data_folder = data_folder + '4class_MI_feedback/data/final_data/'
    # train_file = '%s_%s_%s' % (expInfo['participant'], '4class_MI', expInfo['session'])
    train_file = '%s_%s_%s' % (expInfo['participant'], '2class_MI', expInfo['session'])
    fb_data_file = train_file
    model_file = data_folder + 'models/' + train_file
    model_file = model_file + '_' + str(epochs) + 'epoch_model'

    return


def init_globals_2():
    global subject_name

    subject_name = 'unworn'



def train_network(expInfo):
    init_globals(expInfo)

    # model_file = data_folder + 'models/' + data_file + '_' + str(epochs) + 'epoch_model'

    if clf_method == "LSTM":
        try:
            clf = Deep_Func.load_model(model_file, in_chans=num_channels, input_time_length=buffer_size)
        except FileNotFoundError:
            data_train, label_train, data_val, label_val = get_data()
            clf, opt = get_clf()
            clf = transform_fit(clf, opt, data_train, label_train)
            Deep_Func.save_model(clf, model_file)
    elif clf_method == "Riemann":
        data_train, label_train, data_val, label_val = get_data()
        clf, opt = get_clf()
        # split into training/val set? should determine best model before continuing
        clf = transform_fit(clf, opt, data_train, label_train)

    unique, counts = np.unique(label_val, return_counts=True)
    # print("Labels: ", unique, counts)
    # print(label_val)

    pred_val, cert = predict(clf, data_val, label_val)
    eval_network(label_val, pred_val)

    return clf


def eval_network(label_val, pred_val):
    # plt.hist(label_val.flatten(), bins=1000)
    # plt.show()

    unique, counts = np.unique(label_val, return_counts=True)
    print("Labels: ", unique, counts)
    print(label_val)
    unique, counts = np.unique(pred_val, return_counts=True)
    print("Predicted: ", unique, counts)
    print(pred_val)

    conf_mat = confusion_matrix(label_val, pred_val)
    print(conf_mat)
    tru_pos, prec_i, recall_i = [], [], []
    for i in range(conf_mat.shape[0]):
        tru_pos.append(conf_mat[i, i])
        prec_i.append(conf_mat[i, i]/np.sum(conf_mat[:, i]).astype(float))
        recall_i.append(conf_mat[i, i]/np.sum(conf_mat[i, :]).astype(float))

    accuracy_val = np.sum(tru_pos).astype(float) / (np.sum(conf_mat)).astype(float)
    print("accuracy: {}".format(accuracy_val))

    precision_tot = np.sum(prec_i)/conf_mat.shape[0]
    print("total precision: {}".format(precision_tot))

    precision_cc = np.sum(prec_i[1:]) / (conf_mat.shape[0]-1)
    print("control class precision: {}".format(precision_cc))

    recall_tot = np.sum(recall_i) / conf_mat.shape[0]
    print("total recall: {}".format(recall_tot))

    recall_cc = np.sum(recall_i[1:]) / (conf_mat.shape[0] - 1)
    print("control class recall: {}".format(recall_cc))

    print("# # # # # # # # # # # # # # # # # # # # # # #")
    print(" ")
    print("# # # # # # # # # # # # # # # # # # # # # # #")

    return


def bci_buffer(iter_num, buffer_size):
    current_data = data_test[iter_num:iter_num + buffer_size]
    # print(current_data.shape)
    return current_data


def bci_buffer_rt(current_data, buffer_size, iter_i):
    iter_n = iter_i * buffer_size
    is_new_data = True
    while is_new_data:
        for [new_data, iter_n] in read_bci_data(iter_n):
            i = buffer_size
            while i > 0:
                if i == buffer_size:
                    # print(new_data)
                    current_data[buffer_size - 1] = new_data
                else:
                    current_data[buffer_size - i - 1] = current_data[buffer_size - i]

                i -= 1
            if iter_n > buffer_size:
                is_new_data = False

    return current_data


def iter_bci_buffer(current_data, iter_n):
    real_time = True
    buffer_size = current_data.shape[0]
    if real_time:
        current_data = bci_buffer_rt(current_data, buffer_size, iter_n)
    else:
        current_data = bci_buffer(iter_n, buffer_size)
        iter_n += 1

    return current_data


def read_bci_data(iter_n):
    # data = data_test[iter_n]
    data, timestamps = data_receiver.receive()
    while data.shape[0] < 1:
        data, timestamps = data_receiver.receive()
    for i in range(data.shape[0]):
        iter_n += 1
        final_data.append(data[i, 0:num_channels])
        data_ts.append(timestamps[i])
        # print(data[i, 0:num_channels])
        # yield data[i, 0:num_channels], iter_n
        yield data[i], iter_n


def read_bci_markers():
    global init_ts_d, init_ts_m

    markers, timestamps = markers_receiver.receive()
    # while markers.shape[0] < 1:
    #     markers = markers_receiver.receive()
    for i in range(markers.shape[0]):
        if len(final_markers) > 0:
            if int(markers[i]) != final_markers[len(final_markers) - 1]:
                # print(int(markers[i]))
                final_markers.append(int(markers[i]))
                markers_ts.append(timestamps[i])
        else:
            # print(int(markers[i]))
            final_markers.append(int(markers[i]))
            markers_ts.append(timestamps[i])

            # On first marker stream pull, synchronize with the data stream
            init_ts_m = timestamps[0]
            data, timestamps = data_receiver.receive()
            while data.shape[0] < 1:
                data, timestamps = data_receiver.receive()
            init_ts_d = timestamps[0]


def sync_streams(init_ts):
    global init_ts_m, init_ts_d

    init_ts_m = init_ts

    data, timestamps = data_receiver.receive()
    while data.shape[0] < 1:
        data, timestamps = data_receiver.receive()
    init_ts_d = timestamps[0]


def update_markers(markers_in, markers_ts_in):
    global final_markers, markers_ts

    final_markers = markers_in
    markers_ts = markers_ts_in


def get_bci_class(bci_iter, clf, num_channels=32):
    filter_rt = False

    buffer_size = int(freq*window_size)
    label = [0]
    if bci_iter == 0:
        global buffer_data
        buffer_data = np.zeros((buffer_size, num_channels))
    buffer_data = iter_bci_buffer(buffer_data, bci_iter)
    # print(buffer_data)
    if filter_rt:
        buffer_data = eeg_io_pp_2.butter_bandpass_filter(buffer_data, 7, 30, freq)

    x1 = eeg_io_pp_2.norm_dataset(buffer_data)
    x1 = x1.reshape(1, x1.shape[0], x1.shape[1])

    try:
        a, cert = predict(clf, x1, label)
    except ValueError:
        a, cert = 0, 0

    if np.isnan(cert[0]):
        cert = 0.2
        time.sleep(0.5)


    print(bci_iter, a, cert)

    # final_data[bci_iter] = x1
    # final_label.append(a)

    return a, cert


def get_bci_class_lsl():
    print('entered this thing!')


def init_receiver():
    global data_receiver, num_channels, final_data, data_ts
    data_receiver = LSLa.lslReceiver('data', True, True)
    num_channels = 32
    final_data, data_ts = [], []
    return


def init_marker_receiver():
    global markers_receiver, final_markers, markers_ts
    markers_receiver = LSLa.lslReceiver('markers', True, True)
    final_markers, markers_ts = [], []
    return


def sync_data():
    global init_ts_d, init_ts_m

    data, timestamp_d = data_receiver.receive()
    markers, timestamp_m = markers_receiver.receive()

    init_ts_d = timestamp_d[0]
    init_ts_m = timestamp_m[0]

    return


def save_data(feedback=False, vrep=False):
    global final_data_array, data_ts, final_markers_array, markers_ts, init_ts_d, init_ts_m, subject_name

    # subject_name = "unworn"
    if vrep:
        subject_name = subject_name + "_vrep"
    if feedback:
        subject_name = subject_name + "_fb"

    final_data_array = np.asarray(final_data)
    data_ts_array = np.asarray(data_ts) - init_ts_d
    data_ts_array = data_ts_array.reshape(data_ts_array.shape[0], 1)
    final_data_array_ts = np.concatenate((data_ts_array, final_data_array), axis=1)
    np.savetxt(subject_name + "_EEG_data.csv", final_data_array_ts, delimiter=",")

    final_markers_array = np.asarray(final_markers)
    markers_ts_array = np.asarray(markers_ts) - init_ts_m
    final_markers_array = final_markers_array.reshape(markers_ts_array.shape[0], 1)
    markers_ts_array = markers_ts_array.reshape(markers_ts_array.shape[0], 1)
    final_markers_array_ts = np.concatenate((markers_ts_array, final_markers_array), axis=1)
    print(final_markers_array_ts)
    np.savetxt(subject_name + "_markers.csv", final_markers_array_ts, delimiter=",")

    return


if __name__ == '__main__':

    clf = train_network()

    data_train, label_train, data_val, label_val = get_data()
    # separate into train/val sets?
    unique, counts = np.unique(label_train, return_counts=True)
    print("Labels (train): ", unique, counts)
    unique, counts = np.unique(label_val, return_counts=True)
    print("Labels (val): ", unique, counts)

    pred_val, cert = predict(clf, data_val, label_val)
    eval_network(label_val, pred_val)

    # data_test, label_test = eeg_io_pp.get_data_2a(dataset_dir + test_file, n_classes, remove_rest=False, training_data=False)

    buffer_size = int(freq * window_size)
    # num_channels = 22
    buffer_data = np.zeros((buffer_size, num_channels))
    iter_num = 0

    data_receiver = LSLa.lslReceiver(True, True)

    while buffer_data.shape[0] == buffer_size:
        buffer_data = iter_bci_buffer(buffer_data, iter_num)
        x1 = eeg_io_pp_2.norm_dataset(buffer_data)
        x1 = x1.reshape(1, x1.shape[0], x1.shape[1])

        try:
            a, cert = predict(clf, x1, [0])
        except ValueError:
            a, cert = 0, 0
        except KeyboardInterrupt:
            data_receiver.clean_up()
        print(iter_num, a, cert)
        iter_num += 1

