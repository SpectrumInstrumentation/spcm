"""
Spectrum Instrumentation GmbH (c)

3_acq_multi.py

Shows a simple Standard mode example using only the few necessary commands

Example for analog recording cards (digitizers) for the the M2p, M4i, M4x and M5i card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
import numpy as np
import matplotlib.pyplot as plt


card : spcm.Card

# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(card_type=spcm.SPCM_TYPE_AI) as card:            # if you want to open the first card of a specific type
    
    # setup card mode
    card.card_mode(spcm.SPC_REC_STD_MULTI) # multiple recording mode
    card.timeout(5000)

    # setup channel 0
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    amplitude_mV = 1000
    channels.amp(amplitude_mV)
    max_sample_value = card.max_sample_value()

    # setup trigger engine
    trigger = spcm.Trigger(card)
    # trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)
    trigger.or_mask(spcm.SPC_TMASK_NONE)
    trigger.ch_or_mask0(spcm.SPC_TMASK0_CH0)
    trigger.ch_mode(0, spcm.SPC_TM_POS)
    trigger.ch_level(0, 0, 4)

    # setup clock engine
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)
    sample_rate = clock.sample_rate(3200e6)

    # setup data transfer
    num_samples = 4096
    samples_per_segment = 2048
    multiple_recording = spcm.Multi(card)
    multiple_recording.memory_size(num_samples)
    multiple_recording.allocate_buffer(samples_per_segment)
    multiple_recording.post_trigger(samples_per_segment // 2)
    multiple_recording.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)


    # wait until the transfer has finished
    try:
        card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_DATA_WAITDMA)

        data = multiple_recording.buffer

        # this is the point to do anything with the data
        # e.g. calculate minimum and maximum of the acquired data
        minimum = np.min(data, axis=-1)
        maximum = np.max(data, axis=-1)
        for segment in range(data.shape[0]):
            # print min and max for all channels in all segments
            print("Segment {}".format(segment))
            for channel in channels:
                print("\tChannel {}".format(channel.index))
                print("\t\tMinimum: {:.3f} mV".format(minimum[segment, channel] / max_sample_value * amplitude_mV))
                print("\t\tMaximum: {:.3f} mV".format(maximum[segment, channel] / max_sample_value * amplitude_mV))
    
        plt.figure()
        for segment in range(data.shape[0]):
            for channel in channels:
                plt.plot(np.arange(samples_per_segment) / sample_rate, data[segment, channel.index, :] / max_sample_value * amplitude_mV, '.', label="Ch {}, Seg {}".format(channel.index, segment))
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude [mV]")
        plt.ylim(-amplitude_mV*1.1, amplitude_mV*1.1)
        plt.legend()
        plt.show()
    except spcm.SpcmTimeout as timeout:
        print("Timeout...")


