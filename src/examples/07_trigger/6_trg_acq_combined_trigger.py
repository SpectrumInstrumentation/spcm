"""
Spectrum Instrumentation GmbH (c)

6_trg_acq_combined_trigger.py

Shows a simple Standard acquisition (recording) mode example combining different trigger sources.

Example for analog recording cards (digitizers) for the the M2p, M4i, M4x and M5i card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
from spcm import units

import numpy as np
import matplotlib.pyplot as plt


card : spcm.Card

with spcm.Card('/dev/spcm1') as card: # if you want to open a specific card
    # do a simple standard setup
    card.card_mode(spcm.SPC_REC_STD_SINGLE)     # single trigger standard mode
    card.timeout(5 * units.s)                     # timeout 5 s

    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL) # clock mode internal PLL
    clock.sample_rate(10 * units.percent)
    
    # setup the channels
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0 | spcm.CHANNEL1) # enable channel 0 and 1
    channels.amp(1000 * units.mV)

    ### Trigger setup section ###
    trigger = spcm.Trigger(card, clock=clock)

    selected_trigger_mode = input("There are several trigger modes available. Please select one of the following modes by entering the corresponding number and press <ENTER>:\n" \
    "1: Trigger on positive edge of channel signal, while the external trigger signal is high.\n" \
    "2: Trigger on positive edge of either channel signal.\n"
    )

    try:
        tm = int(selected_trigger_mode)
    except ValueError:
        print("Invalid input. Defaulting to trigger on positive edge of channel signal.")
        tm = 1
    if tm == 1:
        # Trigger when a signal on the channel 0 crosses 500 mV while at the same time the external trigger signal is high
        trigger.and_mask(spcm.SPC_TMASK_EXT0)
        trigger.ext0_mode(spcm.SPC_TM_HIGH)  # external trigger 0 is high
        trigger.ext0_level0(1.0 * units.V)    # external trigger level for
        trigger.ch_and_mask0(channels[0].ch_mask())
        trigger.ch_mode(channels[0], spcm.SPC_TM_POS) # trigger on positive edge of channel 0
        trigger.ch_level0(channels[0], 0.5 * units.V)  # trigger level for channel 0
    elif tm == 2:
        # Trigger when a signal on one of the channels crosses 500 mV
        trigger.ch_or_mask0(channels[0].ch_mask() | channels[1].ch_mask())
        trigger.ch_mode(channels[0], spcm.SPC_TM_POS)  # trigger on positive edge of channel 0
        trigger.ch_level0(channels[0], 0.5 * units.V)  # trigger level for channel 0
        trigger.ch_mode(channels[1], spcm.SPC_TM_POS)  # trigger on positive edge of channel 1
        trigger.ch_level0(channels[1], 0.5 * units.V)  # trigger level for channel 1
    else:
        print("Invalid input. Stopping execution.")
        exit()
    #############################

    # define the data buffer
    data_transfer = spcm.DataTransfer(card)
    num_samples = 512 * units.S
    data_transfer.memory_size(num_samples)
    data_transfer.allocate_buffer(num_samples)
    data_transfer.post_trigger(num_samples // 2)
    
    # start card and wait until recording is finished
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_WAITREADY)

    print("Finished acquiring...")

    # Start DMA transfer and wait until the data is transferred
    data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA, spcm.M2CMD_DATA_WAITDMA)

    # Plot the acquired data
    time_data_s = data_transfer.time_data()
    fig, ax = plt.subplots()
    for channel in channels:
        unit_data_V = channel.convert_data(data_transfer.buffer[channel, :], units.V)
        print(channel)
        print("\tMinimum: {:.3~P}".format(np.min(unit_data_V)))
        print("\tMaximum: {:.3~P}".format(np.max(unit_data_V)))
        ax.plot(time_data_s, unit_data_V, label=f"{channel}")
    ax.yaxis.set_units(units.mV)
    ax.xaxis.set_units(units.us)
    ax.axvline(0, color='k', linestyle='--', label='Trigger')
    ax.legend()
    plt.show()
