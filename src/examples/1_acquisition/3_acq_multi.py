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

    # setup trigger engine
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)

    # setup clock engine
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)
    sample_rate = clock.sample_rate(20e6)

    # setup channel 0
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    amplitude_mV = 1000
    channels.amp(amplitude_mV)
    max_sample_value = card.max_sample_value()

    # setup data transfer
    num_samples = 1024
    samples_per_segment = 256
    num_segments = num_samples // samples_per_segment
    multiple_recording = spcm.Multi(card)
    multiple_recording.memory_size(samples_per_segment*num_segments)
    multiple_recording.allocate_buffer(segment_samples=samples_per_segment, num_segments=num_segments)
    multiple_recording.post_trigger(samples_per_segment // 2)
    multiple_recording.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)

    # wait until the transfer has finished
    try:
        card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_DATA_WAITDMA)

        # this is the point to do anything with the data
        # e.g. calculate minimum and maximum of the acquired data
        minimum = np.min(multiple_recording.buffer, axis=2)
        maximum = np.max(multiple_recording.buffer, axis=2)
        for segment in range(num_segments):
            # print min and max for all channels in all segments
            print("Segment {}".format(segment))
            for channel in channels:
                print("\tChannel {}".format(channel.index))
                print("\t\tMinimum: {:.3f} mV".format(minimum[channel.index, segment] / max_sample_value * amplitude_mV))
                print("\t\tMaximum: {:.3f} mV".format(maximum[channel.index, segment] / max_sample_value * amplitude_mV))
        plt.figure()
        for channel in channels:
            for segment in range(num_segments):
                plt.plot(np.arange(samples_per_segment) / sample_rate, multiple_recording.buffer[channel.index, segment, :] / max_sample_value * amplitude_mV, '.', label="Ch {}, Seg {}".format(channel.index, segment))
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude [mV]")
        plt.legend()
        plt.show()
    except spcm.SpcmTimeout as timeout:
        print("Timeout...")


