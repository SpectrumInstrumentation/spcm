"""  
Spectrum Instrumentation GmbH (c) 2024

9_dds_external_trigger.py

External Trigger - wait for trigger than turn-on carrier 0 and afterwards turn it off again

Example for analog replay cards (AWG) for the the M4i and M4x card-families with installed DDS option.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm

card : spcm.Card
# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(card_type=spcm.SPCM_TYPE_AO) as card:            # if you want to open the first card of a specific type

    # setup card for DDS
    card.card_mode(spcm.SPC_REP_STD_DDS)

    # Setup the card
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    channels.enable(True)
    channels.amp(1000) # 1000 mV
    
    # Activate external trigger mode
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_EXT0) # disable default software trigger
    trigger.ext0_mode(spcm.SPC_TM_POS) # positive edge
    trigger.ext0_level0(1500) # Trigger level is 1.5 V (1500 mV)
    trigger.ext0_coupling(spcm.COUPLING_DC) # set DC coupling
    card.write_setup()
    
    # Setup DDS
    dds = spcm.DDS(card)
    dds.reset()

    # Start the DDS test
    dds.trg_src(spcm.SPCM_DDS_TRG_SRC_CARD)

    # Create one carrier and keep it off
    dds.amp(0, 0.4)
    dds.freq(0, 5e6) # 5 MHz
    dds.exec_at_trg()

    # each trigger event will change the generated frequency by 1 MHz
    for i in range(1, 11):
        dds.freq(0, 5e6 + i*1e6)
        dds.exec_at_trg()    # turn off as soon as possible again
    dds.write_to_card()

    # Start command including enable of trigger engine
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_FORCETRIGGER)

    input("Press Enter to Exit")

