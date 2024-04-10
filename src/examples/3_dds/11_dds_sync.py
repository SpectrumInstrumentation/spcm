"""  
Spectrum Instrumentation GmbH (c) 2024

11_dds_starhub_sync.py

Output a single carrier on two cards synced through a StarHub - This example shows the DDS functionality with 1 carrier with a fixed frequency and fixed amplitude.

Example for analog replay cards (AWG) for the the M4i and M4x card-families with installed DDS option.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
from spcm import units


# Load the cards
card_identifiers = ["/dev/spcm0", "/dev/spcm1"]
sync_identifier  = "sync0"

# open cards and sync
stack : spcm.CardStack
with spcm.CardStack(card_identifiers=card_identifiers, sync_identifier=sync_identifier) as stack:
    # setup all the channels in the card stack
    channels = spcm.Channels(stack=stack, stack_enable=[spcm.CHANNEL0, spcm.CHANNEL0])
    channels.enable(True)
    channels.amp(1 * units.V)

    multi_dds = []
    for card in stack.cards:

        # setup card for DDS
        card.card_mode(spcm.SPC_REP_STD_DDS)
        card.write_setup()
        
        # Setup DDS
        dds = spcm.DDS(card)
        multi_dds.append(dds)
        dds.reset()

        # Start the test
        dds[0].amp(40 * units.percent)
        dds[0].freq(1.0 * units.MHz)
        dds.trg_src(spcm.SPCM_DDS_TRG_SRC_NONE)
        dds.exec_at_trg()
        dds.write_to_card()
    
    # setup synchronization between the cards
    stack.sync_enable(True)

    stack.start(spcm.M2CMD_CARD_ENABLETRIGGER)
    
    stack.force_trigger()

    input("Press Enter to Exit")
