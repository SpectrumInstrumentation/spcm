"""
Spectrum Instrumentation GmbH (c)

5_acq_single_digital.py

Shows a simple Standard mode example using only the few necessary commands to test digital acquisition

Example for digital recording cards for the the M2p, M4i, M4x and M5i card-families.

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
with spcm.Card(card_type=(spcm.SPCM_TYPE_DIO | spcm.SPCM_TYPE_DI)) as card:            # if you want to open the first card of a specific type
    
    # do a simple standard setup
    card.card_mode(spcm.SPC_REC_STD_SINGLE)     # standard single acquisition mode
    card.timeout(5 * units.s)

    # setup the channels
    channels = spcm.Channels(card, card_enable=0xFFFF)
    channels.digital_termination(0, True)

    # Set the trigger to software trigger
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)

    # setup clock engine
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)
    clock.sample_rate(max=True)

    # define the data buffer
    num_samples = 16 * units.KiS # KibiSamples = 1024 Samples
    data_transfer = spcm.DataTransfer(card)
    data_transfer.memory_size(num_samples)
    data_transfer.allocate_buffer(num_samples)
    data_transfer.post_trigger(num_samples // 2)
    data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA, direction=spcm.SPCM_DIR_CARDTOPC)

    # start everything
    try:
        card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_DATA_WAITDMA)
    except spcm.SpcmException as exception:
        print("... Timeout")
    
    data_transfer.unpackbits()

    # Plot the acquired data
    time_data = data_transfer.time_data()
    fig, ax = plt.subplots(len(channels), 1, sharex=True)
    for channel in channels:
        ax[channel].step(time_data, data_transfer.bit_buffer[:, channel], label=f"{channel.index}")
        ax[channel].set_ylabel(f"{channel.index}")
        ax[channel].set_yticks([])
        ax[channel].set_ylim([-0.1, 1.1])
        ax[channel].xaxis.set_units(units.us)
    fig.text(0.04, 0.5, 'channel', va='center', rotation='vertical')
    plt.show()

    print("Finished...")


