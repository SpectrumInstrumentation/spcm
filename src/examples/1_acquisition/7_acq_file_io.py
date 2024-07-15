"""
Spectrum Instrumentation GmbH (c)

7_acq_file_io.py

Shows a simple Standard mode example acquisition example with the addition of writing the acquired data to a file and reading it back.
- connect a function generator that generates a sine wave with 10-100 kHz frequency and 100 - 1000 mV amplitude to channel 0

Example for analog recording cards (digitizers) for the the M2p, M4i, M4x and M5i card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
from spcm import units

import matplotlib.pyplot as plt

with spcm.Card(card_type=spcm.SPCM_TYPE_AI) as card:
    # do a simple standard setup
    card.card_mode(spcm.SPC_REC_STD_SINGLE)     # single trigger standard mode
    card.timeout(5 * units.s)

    clock = spcm.Clock(card)
    clock.sample_rate(20 * units.MHz)
    
    # setup the channels
    channels = spcm.Channels(card) # enable all channels
    channels.amp(1 * units.V)

    # define the data buffer
    num_samples = 1 * units.KiS
    data_transfer = spcm.DataTransfer(card)
    data_transfer.memory_size(num_samples)
    data_transfer.allocate_buffer(num_samples)
    # Start DMA transfer
    data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)
    
    # start card
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_DATA_WAITDMA)

    # Write to file
    print("Writing data to file: data.*")
    data_transfer.tofile('data.csv', delimiter=',')
    # data_transfer.tofile('data.hdf5')
    # data_transfer.tofile('data.bin')

    # Read back from file
    print("Reading data from file: data.*")
    data_transfer.fromfile('data.csv', delimiter=',')
    # data_transfer.fromfile('data.hdf5')
    # data_transfer.fromfile('data.bin', dtype=np.int8, shape=(len(channels), num_samples))

    fig, ax = plt.subplots()
    time_data = data_transfer.time_data()
    for channel in channels:
        data = channel.convert_data(data_transfer.buffer[channel], units.V)
        ax.plot(time_data, data, label=f"{channel.index}")
    ax.xaxis.set_units(units.ms)
    ax.yaxis.set_units(units.mV)
    ax.legend()
    plt.show()