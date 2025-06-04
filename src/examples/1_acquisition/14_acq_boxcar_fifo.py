"""
Spectrum Instrumentation GmbH (c)

14_acq_boxcar_fifo.py

Shows a simple FIFO mode with boxcar averaging example using only the few necessary commands
- connect a function generator that generates a sine wave with 10-100 kHz frequency and 200 mV amplitude to channel 0

Example for analog recording cards (digitizers) for the the M2p, M4i, M4x and M5i card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
from spcm import units

import numpy as np
import matplotlib.pyplot as plt
        

card : spcm.Card

# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(card_type=spcm.SPCM_TYPE_AI, verbose=False) as card:            # if you want to open the first card of a specific type
    
    # single FIFO mode
    card.card_mode(spcm.SPC_REC_FIFO_BOXCAR)
    card.timeout(5 * units.s)

    # setup trigger engine
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_NONE)
    trigger.and_mask(spcm.SPC_TMASK_NONE)

    # setup clock engine
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)
    sampling_rate = clock.sample_rate(max=True, return_unit=units.MHz) # Adjusted sample rate
    print(f"Sampling rate: {sampling_rate}")

    # setup channels
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    channels.amp(1 * units.V)
    channels.termination(1)

    # Channel triggering
    trigger.ch_or_mask0(channels[0].ch_mask())
    trigger.ch_mode(channels[0], spcm.SPC_TM_POS)
    trigger.ch_level0(channels[0], 0 * units.mV, return_unit=units.mV)

    # define the data buffer
    notify_samples = 8 * units.KiS
    samples_per_segment = 1 * units.KiS
    num_segments = 16
    averages = 8
    plot_segments = 4

    num_samples = samples_per_segment * num_segments
    post_trigger = samples_per_segment - 128 * units.S

    boxcar = spcm.Boxcar(card)
    boxcar.box_averages(averages)  # Set boxcar averaging factor
    boxcar.allocate_buffer(segment_samples=samples_per_segment, num_segments=num_segments)
    boxcar.notify_samples(notify_samples)
    boxcar.post_trigger(post_trigger)

    boxcar.start_buffer_transfer()
    boxcar.verbose(True)

    # start the card
    card.start(spcm.M2CMD_DATA_STARTDMA | spcm.M2CMD_CARD_ENABLETRIGGER)

    data_array = np.array([])
    try:
        print("Press Ctrl+C to stop the recording and show the results...")
        # Get a block of data
        for data_block in boxcar:
            if data_array.size == 0:
                data_array = data_block
            else:
                data_array = np.append(data_array, data_block, axis=0)
    except KeyboardInterrupt as e:
        pass

    # Print the results
    print("Finished...\n")
    
    # Plot the accumulated data
    time_data_s = boxcar.time_data(total_num_samples=samples_per_segment)
    fig, ax = plt.subplots()
    for channel in channels:
        print(channel)
        for segment_index in range(plot_segments):
            print(f"\tSegment {segment_index}:")
            unit_data_V = channel.convert_data(data_array[segment_index, :, channel], units.V, averages=averages)
            print("\t\tMinimum: {:.3~P}".format(np.min(unit_data_V)))
            print("\t\tMaximum: {:.3~P}".format(np.max(unit_data_V)))
            ax.plot(time_data_s, unit_data_V, label=f"{channel} - Segment {segment_index}")
    ax.xaxis.set_units(units.ms)
    ax.yaxis.set_units(units.mV)
    ax.axvline(0, color='k', linestyle='--', label='Trigger')
    ax.legend()
    plt.show()

