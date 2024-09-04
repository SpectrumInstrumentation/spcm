"""
Spectrum Instrumentation GmbH (c)

4_gen_fifo.py

Shows a simple FIFO mode example using only the few necessary commands
- output on channel 0 and 1
- 10% of maximum sample rate of the card
- channel 0: sine wave with 40 kHz frequency and 1 V amplitude
- channel 1: sine wave with 20 kHz frequency and 1 V amplitude

Example for analog replay cards (AWG) for the the M2p, M4i, M4x and M5i card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
from spcm import units

import numpy as np

# to speed up the calculation of new data we pre-calculate the signals
# to simplify that we use special frequencies
signal_frequency = np.array([ 40000, 20000 ]) * units.Hz

card : spcm.Card
# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(card_type=spcm.SPCM_TYPE_AO) as card:             # if you want to open the first card of a specific type
    
    # set up the mode
    card.card_mode(spcm.SPC_REP_FIFO_SINGLE)

    # setup channels 0 and 1
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0 | spcm.CHANNEL1)
    channels.enable(True)
    channels.output_load(units.highZ)
    channels.amp(1 * units.V)

    # set samplerate to 10% of the maximum, no clock output
    clock = spcm.Clock(card)
    sample_rate = clock.sample_rate(10 * units.percent, return_unit=units.MHz) # 10% of the maximum sample rate and returns the sampling rate in MHz
    clock.clock_output(False)

    # setup the trigger mode
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)

    # define the data buffer
    num_samples = 8 * units.MiS
    notify_samples = 512 * units.KiS

    data_transfer = spcm.DataTransfer(card)
    data_transfer.memory_size(num_samples) # size of memory on the card
    data_transfer.allocate_buffer(num_samples) # size of buffer in pc RAM
    data_transfer.pre_trigger(1024 * units.S)
    data_transfer.notify_samples(notify_samples)

    ############################
    print("Pre-calculate the data...")
    min_freq = np.min(signal_frequency[:len(channels)])
    num_data = int((sample_rate / min_freq).to_base_units().magnitude)
    
    time_data = data_transfer.time_data(num_data)
    data_matrix = np.empty((len(channels), num_data), dtype=np.int16)
    for channel in channels:
        data_matrix[channel.index, :] = np.int16(32767 * np.sin(2.* np.pi*time_data * signal_frequency[channel.index]).to(units.fraction).magnitude)
    
    ############################
    print("Pre-fill the complete DMA buffer...")
    memory_indices = np.mod(np.arange(num_samples.to_base_units().magnitude), num_data)
    data_transfer.buffer[:, memory_indices] = data_matrix[:, memory_indices]
    current_sample_position = int(num_samples.to(units.S).magnitude)
    notify_samples_mag = int(notify_samples.to(units.S).magnitude)

    # we define the buffer for transfer and start the DMA transfer
    data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)
    data_transfer.avail_card_len(num_samples)

    # pre-fill the data buffer
    for data_block in data_transfer:
        data_indices = np.mod(np.arange(current_sample_position, current_sample_position + notify_samples_mag), num_data)
        data_block[:] = data_matrix[:, data_indices] # !!! data_block is a numpy ndarray and you need to write into that array, hence the [:]
        current_sample_position += notify_samples_mag
        fill_size = data_transfer.fill_size_promille()
        print("Fill size: {}".format(fill_size), end="\r")
        if fill_size == 1000:
            break
    print("... data has been transferred to board memory")

    ############################
    print("Starting the card...")
    print("press Ctrl+C to stop the generation of the signals")
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER)
    # We'll start the replay and run until a timeout occurs or user interrupts the program
    try:
        for data_block in data_transfer:
            data_indices = np.mod(np.arange(current_sample_position, current_sample_position + notify_samples_mag), num_data)
            data_block[:] = data_matrix[:, data_indices] # !!! data_block is a numpy ndarray and you need to write into that array, hence the [:]
            current_sample_position += notify_samples_mag
    except spcm.SpcmException as exception:
        # Probably a buffer underrun has happened, capure the event here
        print(exception)
    except KeyboardInterrupt:
        print("\n... interrupted by user")


