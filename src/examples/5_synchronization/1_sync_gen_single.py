"""  
Spectrum Instrumentation GmbH (c) 2024

1_sync_gen_single.py

Shows a simple standard mode example with a StarHub and two cards.
On the first card there is a sine wave on each channel, on the second card there is
a saw-tooth at each channel.

Example for analog replay cards (AWG) for the the M2p, M4i and M4x card-families with Starhub synchronization.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
from spcm import units

import numpy as np

# Load the cards
card_identifiers = ["/dev/spcm0", "/dev/spcm1"]
sync_identifier  = "sync0"

# open cards and sync
stack : spcm.CardStack
with spcm.CardStack(card_identifiers=card_identifiers, sync_identifier=sync_identifier) as stack:

    # setup all the channels
    channels = spcm.Channels(stack=stack)
    channels.enable(True)
    channels.amp(1 * units.V)

    master = 0
    for i, card in enumerate(stack.cards):
        # read function and sn and check for D/A card
        if card.function_type() != spcm.SPCM_TYPE_AO:
            spcm.SpcmException(f"This is an example for D/A cards.\n{card} not supported by this example\n")
        print(f"Found: {card}")

        # set up the mode
        card.card_mode(spcm.SPC_REP_STD_CONTINUOUS)
        card.loops(0) # loop continuously

        # setup the clock
        clock = spcm.Clock(card)
        sample_rate = clock.sample_rate(spcm.MEGA(50))
        clock.clock_output(False)

        # setup the trigger mode
        trigger = spcm.Trigger(card)
        features = card.features()
        if features & (spcm.SPCM_FEAT_STARHUB5 | spcm.SPCM_FEAT_STARHUB16):
            # set star-hub carrier card as clock master and trigger master
            trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)
            master = i
        else:
            trigger.or_mask(spcm.SPC_TMASK_NONE)
        trigger.and_mask(0)

    # setup synchronization between the cards
    stack.sync_enable(True)

    # do a simple setup in CONTINUOUS replay mode for each card
    num_samples = spcm.KIBI(64)
    for i, card in enumerate(stack.cards):
        # setup software buffer
        data_transfer = spcm.DataTransfer(card)
        data_transfer.memory_size(num_samples)
        data_transfer.allocate_buffer(num_samples)

        card_channels = spcm.Channels(card)

        max_value = card.max_sample_value()
        max_value = max_value - 1

        sample_space = np.arange(-num_samples/2, num_samples/2)

        # calculate the data
        if i == 0: 
            # first card, generate a sine on each channel
            for channel in card_channels:
                factor = np.sin(2 * np.pi * sample_space / (num_samples / (channel + 1)))
                data_transfer.buffer[channel, :] = np.int16(max_value * factor)
        elif i == 1:
            # second card, generate a rising ramp on each channel
            for channel in card_channels:
                factor = sample_space / (num_samples / (channel + 1))
                data_transfer.buffer[channel, :] = np.int16(max_value * factor)
            
        # we define the buffer for transfer and start the DMA transfer
        data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)

        card.timeout(10000)


    # We'll start and wait until the card has finished or until a timeout occurs
    # since the card is running in SPC_REP_STD_CONTINUOUS mode with SPC_LOOPS = 0 we will see the timeout
    try:
        stack.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_WAITREADY)
    except spcm.SpcmTimeout as timeout:
        print("Finished outputting")
        stack.stop()

