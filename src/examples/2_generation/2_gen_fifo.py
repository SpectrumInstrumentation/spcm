"""
Spectrum Instrumentation GmbH (c)

2_gen_fifo.py

Shows a simple FIFO mode example using only the few necessary commands

Example for analog replay cards (AWG) for the the M2p, M4i and M4x card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
import numpy as np

# to speed up the calculation of new data we pre-calculate the signals
# to simplify that we use special frequencies
signal_frequency_Hz = [ 40000, 20000, 10000, 5000, 2500, 1250, 625, 312.5 ]

card : spcm.Card
# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(card_type=spcm.SPCM_TYPE_AO) as card:             # if you want to open the first card of a specific type
    
    # set up the mode
    card.card_mode(spcm.SPC_REP_FIFO_SINGLE)

    # setup all channels
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0 | spcm.CHANNEL1)
    channels.enable(True)
    channels.amp(1000) # 1000 mV

    # set samplerate to 50 MHz (M4i) or 1 MHz (otherwise), no clock output
    clock = spcm.Clock(card)
    series = card.series()
    if (series in [spcm.TYP_M4IEXPSERIES, spcm.TYP_M4XEXPSERIES]):
        sample_rate = clock.sample_rate(spcm.MEGA(50))
    else:
        sample_rate = clock.sample_rate(spcm.MEGA(1))
    clock.clock_output(0)

    # setup the trigger mode
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)

    # define the data buffer
    num_samples = spcm.MEBI(8)
    notify_samples = spcm.KIBI(512)

    data_transfer = spcm.DataTransfer(card)
    data_transfer.memory_size(num_samples)
    data_transfer.allocate_buffer(num_samples)
    data_transfer.pre_trigger(1024)
    data_transfer.notify_samples(notify_samples)

    # Precalculating the data
    num_data = int(sample_rate / np.min(signal_frequency_Hz[:len(channels)]))
    
    data_range = np.arange(num_data)
    data_matrix = np.empty((len(channels), num_data), dtype=np.int16)
    for channel in channels:
        data_matrix[channel.index, :] = np.int16(32767 * np.sin(2.* np.pi*data_range/(sample_rate / signal_frequency_Hz[channel.index])))
    
    # pre-fill the complete DMA buffer
    memory_indices = np.mod(np.arange(num_samples), num_data)
    data_transfer.buffer[:, memory_indices] = data_matrix[:, memory_indices]
    current_sample_position = num_samples

    # we define the buffer for transfer and start the DMA transfer
    data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA, direction=spcm.SPCM_DIR_PCTOCARD)
    data_transfer.avail_card_len(num_samples)

    # pre-fill the data buffer
    for data_block in data_transfer:
        data_indices = np.mod(np.arange(current_sample_position, current_sample_position + notify_samples), num_data)
        data_block[:] = data_matrix[:, data_indices] # !!! data_block is a numpy ndarray and you need to write into that array, hence the [:]
        current_sample_position += notify_samples
        if data_transfer.fill_size_promille() == 1000:
            break
    print("\n... data has been transferred to board memory")
    print("Starting the card...")
    print("press Ctrl+C to stop the example")
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER)
    # We'll start the replay and run until a timeout occurs or user interrupts the program
    try:
        for data_block in data_transfer:
            data_indices = np.mod(np.arange(current_sample_position, current_sample_position + notify_samples), num_data)
            data_block[:] = data_matrix[:, data_indices] # !!! data_block is a numpy ndarray and you need to write into that array, hence the [:]
            current_sample_position += notify_samples
    except spcm.SpcmException as exception:
        # Probably a buffer underrun has happened, capure the event here
        print(exception)
    except KeyboardInterrupt:
        print("\n... interrupted by user")


