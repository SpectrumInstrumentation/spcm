"""
Spectrum Instrumentation GmbH (c)

4_acq_single_digital.py

Shows a simple Standard mode example using only the few necessary commands to test digital acquisition

Example for digital recording cards for the the M2p, M4i, M4x and M5i card-families.

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
with spcm.Card(card_type=(spcm.SPCM_TYPE_DIO | spcm.SPCM_TYPE_DI)) as card:            # if you want to open the first card of a specific type
    
    # do a simple standard setup
    # num_channels = card.channels_enable(0xFFFFFFFF) # 32 bits enabled
    card.card_mode(spcm.SPC_REC_STD_SINGLE)     # standard single acquisition mode
    card.timeout(5000)

    # setup the channels
    channels = spcm.Channels(card, card_enable=0xFFFFFFFF)

    # Set the trigger to software trigger
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)
    trigger.and_mask(spcm.SPC_TMASK_NONE)

    # setup clock engine
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)
    sample_rate = clock.sample_rate(spcm.MEGA(125))
    clock.output(0)

    # define the data buffer
    num_samples = spcm.KIBI(16)
    data_transfer = spcm.DataTransfer(card)
    data_transfer.post_trigger(num_samples // 2)
    data_transfer.memory_size(num_samples)
    data_transfer.allocate_buffer(num_samples)
    data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA, direction=spcm.SPCM_DIR_CARDTOPC)

    # start everything
    try:
        card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_DATA_WAITDMA)
    except spcm.SpcmException as exception:
        print("... Timeout")

    # this is the point to do anything with the data
    # e.g. print first 100 samples to screen
    # for sample in data_transfer.buffer[:100]:
    #     print("0b{:032b}".format(sample))

    # Plot the acquired data
    x_axis = np.arange(num_samples)/sample_rate
    plt.figure()
    plt.plot(x_axis, data_transfer.buffer)
    plt.show()

    print("Finished...")


