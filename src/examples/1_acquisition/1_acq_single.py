"""
Spectrum Instrumentation GmbH (c)

1_acq_single.py

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
    
    # do a simple standard setup
    card.card_mode(spcm.SPC_REC_STD_SINGLE)     # single trigger standard mode
    card.timeout(5000)                     # timeout 5 s

    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)       # trigger set to software
    trigger.and_mask(0)

    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)            # clock mode internal PLL
    # we'll try to set the samplerate to 20 MHz
    sample_rate = clock.sample_rate(spcm.MEGA(20))
    
    # setup the channels
    channels = spcm.Channels(card) # enable all channels
    amplitude_mV = 1000
    channels.amp(amplitude_mV)
    channels.termination(1)
    max_sample_value = card.max_sample_value()

    # define the data buffer
    num_samples = spcm.KIBI(1)
    data_transfer = spcm.DataTransfer(card)
    data_transfer.memory_size(num_samples)
    data_transfer.allocate_buffer(num_samples)
    data_transfer.post_trigger(num_samples // 2) # half of the total number of samples after trigger event
    # Start DMA transfer
    data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)
    
    # start card
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_DATA_WAITDMA)

    # The extrema of the data
    minimum = np.min(data_transfer.buffer, axis=1)
    maximum = np.max(data_transfer.buffer, axis=1)

    print("Finished...\n")
    for channel in channels:
        print("Channel {}".format(channel.index))
        print("\tMinimum: {}".format(minimum[channel.index]))
        print("\tMaximum: {}".format(maximum[channel.index]))

    # Plot the acquired data
    x_axis = np.arange(num_samples)/sample_rate
    plt.figure()
    for channel in channels:
        plt.plot(x_axis, data_transfer.buffer[channel.index, :] / max_sample_value * amplitude_mV, label=f"Channel {channel.index}")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [mV]")
    plt.legend()
    plt.show()
