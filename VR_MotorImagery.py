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


def signal_handler():
    print('Program Ending')
    exit()


if __name__ == '__main__':
    freq = 250
    window_size = 0.5

    num_channels = 32
    buffer_size = int(freq*window_size)

    feedback = True
    experiment_going = True  # How to indicate when the experiment has stopped?

    real_time_BCI.init_receiver()

    if feedback:
        expInfo = {'participant': 'daniel', 'session': '001'}

        info_fb = StreamInfo(name='feedbackStream', type='LSL_fb_class', channel_count=1, channel_format='int8')
        outlet_fb = StreamOutlet(info_fb)
        info_cert = StreamInfo(name='certaintyStream', type='LSL_fb_cert', channel_count=1, channel_format='float32')
        outlet_cert = StreamOutlet(info_cert)

        clf = real_time_BCI.train_network(expInfo)      # need to send information about file

    buffer_data = np.zeros((buffer_size, num_channels))
    iter_n = 0

    atexit.register(real_time_BCI.save_data)

    t_start = time.clock()

    while(experiment_going):
        label = []
        buffer_data = real_time_BCI.iter_bci_buffer(buffer_data, iter_n)
        real_time_BCI.read_bci_markers()

        if feedback:
            x1 = eeg_io_pp_2.norm_dataset(buffer_data)
            x1 = x1.reshape(1, x1.shape[0], x1.shape[1])

            try:
                a, cert = real_time_BCI.predict(clf, x1, label)
            except ValueError:
                a, cert = [0], [0]

            outlet_fb.push_sample(a)
            outlet_cert.push_sample(cert)

        if time.clock() > t_start + 100:
            exit()

