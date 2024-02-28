"""
Spectrum Instrumentation GmbH (c)

2_acq_fifo.py

Shows a simple FIFO mode example using only the few necessary commands

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
    
    # single FIFO mode
    card.card_mode(spcm.SPC_REC_FIFO_SINGLE)
    card.timeout(5000)

    # setup trigger engine
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)

    # setup clock engine
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)
    sample_rate = clock.sample_rate(spcm.MEGA(20))

    # setup channels 0 and 1
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0 | spcm.CHANNEL1)
    amplitude_mV = 1000
    channels.amp(amplitude_mV)
    channels.termination(1)
    max_value = card.max_sample_value()

    # define the data buffer
    num_samples = spcm.KIBI(512)
    notify_samples = spcm.KIBI(128)
    plot_samples = spcm.KIBI(1)

    data_transfer = spcm.DataTransfer(card)
    data_transfer.memory_size(num_samples)
    data_transfer.allocate_buffer(num_samples)
    data_transfer.pre_trigger(1024)
    data_transfer.notify_samples(notify_samples)
    data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)

    # start the card
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER)

    try:
        print("Press Ctrl+C to stop the example...")
        # Get the first data block
        data_array = next(data_transfer)
        for data_block in data_transfer:
            data_array = np.append(data_array, data_block, axis=1)
    except KeyboardInterrupt as e:
        pass

    # Print the results
    print("\nFinished...")
    minimum = np.min(data_array)
    maximum = np.max(data_array)
    print("Minimum: {0:d}".format(minimum))
    print("Maximum: {0:d}".format(maximum))
    
    # Plot the accumulated data
    plt.figure()
    for channel in channels:
        plt.plot(np.arange(plot_samples)/sample_rate, data_array[channel.index,:plot_samples]/max_value*amplitude_mV, label="Channel {}".format(channel.index))
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [mV]")
    plt.legend()
    plt.show()

