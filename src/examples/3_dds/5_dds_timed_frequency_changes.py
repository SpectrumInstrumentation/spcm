"""  
Spectrum Instrumentation GmbH (c) 2024

5_dds_timed_frequency_changes.py

Single static carrier with a frequency change after time - This example shows the DDS functionality with 1 carrier with a 100 MHz 
frequency, that changes to 200 MHz after a time of 3 seconds and then again to 300 MHz after 3 seconds.

Example for analog replay cards (AWG) for the the M4i and M4x card-families with installed DDS option.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
from spcm import units


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
    channels.output_load(50 * units.ohm)
    channels.amp(1 * units.V)
    card.write_setup()
    
    # Setup DDS
    dds = spcm.DDS(card, channels=channels)
    dds.reset()

    # Start the test
    dds.trg_src(spcm.SPCM_DDS_TRG_SRC_TIMER)
    dds.trg_timer(3.0 * units.s)
    dds[0].amp(40 * units.percent)
    dds[0].freq(5 * units.MHz)
    dds.exec_at_trg()
    
    dds[0].freq(10 * units.MHz)
    dds.exec_at_trg()
    
    dds[0].freq(15 * units.MHz)
    dds.exec_at_trg()
    
    dds.write_to_card()

    # Start command including enable of trigger engine
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_FORCETRIGGER)

    input("Press Enter to Exit")
