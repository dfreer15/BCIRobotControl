import real_time_BCI_train as real_time_BCI
import eeg_io_pp_2
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream
import numpy as np
import atexit
import signal

import time

# import threading
# import OnlineExp
# import procEEG


if __name__ == '__main__':
    freq = 250
    window_size = 0.5

    num_channels = 32
    buffer_size = int(freq*window_size)

    feedback = True
    experiment_going = True  # How to indicate when the experiment has stopped?
    filter_rt = False

    real_time_BCI.init_globals_2()
    real_time_BCI.get_test_data()  # if using prerecorded data
    real_time_BCI.init_receiver()
    real_time_BCI.init_marker_receiver()

    if feedback:
        expInfo = {'participant': 'daniel', 'session': '001'}
        clf = real_time_BCI.train_network(expInfo)      # need to send information about file

    info_fb = StreamInfo(name='feedbackStream', type='LSL_fb_class', channel_count=1, channel_format='int8')
    outlet_fb = StreamOutlet(info_fb)
    info_cert = StreamInfo(name='certaintyStream', type='LSL_fb_cert', channel_count=1, channel_format='float32')
    outlet_cert = StreamOutlet(info_cert)

    buffer_data = np.zeros((buffer_size, num_channels))
    iter_n = 0

    atexit.register(real_time_BCI.save_data, feedback=feedback, vrep=False)

    t_start = time.clock()

    # real_time_BCI.sync_data()

    while(experiment_going):
        label = []
        buffer_data = real_time_BCI.iter_bci_buffer(buffer_data, iter_n)
        real_time_BCI.read_bci_markers()

        if feedback:
            # print(buffer_data)
            if filter_rt:
                buffer_data = eeg_io_pp_2.butter_bandpass_filter(buffer_data, 7, 30, freq)
            x1 = eeg_io_pp_2.norm_dataset(buffer_data)
            x1 = x1.reshape(1, x1.shape[0], x1.shape[1])

            try:
                a, cert = real_time_BCI.predict(clf, x1, label)
            except ValueError:
                a, cert = 1, 0.001

            print(a, cert)
            outlet_fb.push_sample([int(a)])
            outlet_cert.push_sample([cert])

        if time.clock() > t_start + 300:
            exit()

        iter_n += 1

