"""  
Spectrum Instrumentation GmbH (c) 2024

10_dds_xio.py

XIO usage - Turn the x0 line on for 1 second and then off again

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
with spcm.Card(card_type=spcm.SPCM_TYPE_AO) as card:             # if you want to open the first card of a specific type

    # setup card for DDS
    card.card_mode(spcm.SPC_REP_STD_DDS)

    # Setup the card
    channels = spcm.Channels(card)
    channels.enable(True)
    channels.output_load(50 * units.ohm)
    channels.amp(1 * units.V)
    
    # Activate the xio dds mode
    multi_ios = spcm.MultiPurposeIOs(card)
    multi_ios[0].x_mode(spcm.SPCM_XMODE_DDS)
    multi_ios[1].x_mode(spcm.SPCM_XMODE_DDS)
    multi_ios[2].x_mode(spcm.SPCM_XMODE_DDS)
    card.write_setup()
    
    # Setup DDS
    dds = spcm.DDS(card, channels=channels)
    core0 = dds[0]
    dds.reset()

    # Start the DDS test
    dds.trg_src(spcm.SPCM_DDS_TRG_SRC_TIMER)
    dds.trg_timer(1.0 * units.s)

    dds.x_mode(0, spcm.SPCM_DDS_XMODE_MANUAL)
    dds.x_mode(1, spcm.SPCM_DDS_XMODE_WAITING_FOR_TRG)
    dds.x_mode(2, spcm.SPCM_DDS_XMODE_EXEC)
    dds.exec_at_trg()

    # Create one carrier and keep it off
    core0.amp(40 * units.percent)
    core0.freq(10 * units.Hz)
    dds.x_manual_output(spcm.SPCM_DDS_X0)
    dds.exec_at_trg()

    # set all manually controlled XIO lines to LOW
    core0.amp(0.0)
    dds.x_manual_output(0x0)
    dds.exec_at_trg()

    dds.write_to_card()

    # Start command including enable of trigger engine
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_FORCETRIGGER)

    input("Press Enter to Exit")
