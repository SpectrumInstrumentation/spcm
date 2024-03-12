"""
Spectrum Instrumentation GmbH (c)

7_acq_file_io.py

Shows a simple Standard mode example acquisition example with the addition of writing the acquired data to a file and reading it back.

Example for analog recording cards (digitizers) for the the M2p, M4i, M4x and M5i card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
import numpy as np
import matplotlib.pyplot as plt

with spcm.Card(card_type=spcm.SPCM_TYPE_AI) as card:
    # do a simple standard setup
    card.card_mode(spcm.SPC_REC_STD_SINGLE)     # single trigger standard mode
    card.timeout(5000)

    clock = spcm.Clock(card)
    sample_rate = clock.sample_rate(spcm.MEGA(20))
    
    # setup the channels
    channels = spcm.Channels(card) # enable all channels
    amplitude_mV = 1000
    channels.amp(amplitude_mV)

    # define the data buffer
    num_samples = spcm.KIBI(1)
    data_transfer = spcm.DataTransfer(card)
    data_transfer.memory_size(num_samples)
    data_transfer.allocate_buffer(num_samples)
    # Start DMA transfer
    data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)
    
    # start card
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_DATA_WAITDMA)

    # Write to file
    data_transfer.tofile('data.csv', delimiter=',')
    # data_transfer.tofile('data.hdf5')
    # data_transfer.tofile('data.bin')

    # Read back from file
    data_transfer.fromfile('data.csv', delimiter=',')
    # data_transfer.fromfile('data.hdf5')
    # data_transfer.fromfile('data.bin', dtype=np.int8, shape=(len(channels), num_samples))

    plt.figure()
    for channel in data_transfer.buffer:
        plt.plot(channel)
    plt.show()