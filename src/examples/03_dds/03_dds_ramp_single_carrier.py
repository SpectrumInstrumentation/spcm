"""  
Spectrum Instrumentation GmbH (c) 2024

3_dds_ramp_single_carrier.py

Ramp single carrier - Use the ramping functionality to do a long and slow frequency as well as amplitude ramp

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
    card.write_setup() # IMPORTANT! this turns on the card's system clock signals, that are required for DDS to work
    
    # Setup DDS
    dds = spcm.DDS(card, channels=channels)
    dds.reset()

    # Start the DDS test
    # Timer changes every 2.0 seconds
    period_s = 2.0 * units.s
    dds.trg_src(spcm.SPCM_DDS_TRG_SRC_TIMER)
    dds.trg_timer(period_s)
    # For slow ramps only change the value every 1000 steps
    dds.freq_ramp_stepsize(1000)
    dds.amp_ramp_stepsize(1000)

    # Create one carrier and keep on for 2 seconds
    dds[0].amp(40 * units.percent)
    dds[0].freq(5 * units.MHz) # 5 MHz
    dds.exec_at_trg()

    # Ramp the frequency of the carrier
    dds[0].frequency_slope(5 * units.MHz / units.s) # 5 MHz/s
    dds.exec_at_trg()

    # Stop frequency ramp
    dds[0].frequency_slope(0)
    dds[0].freq(15 * units.MHz) # 15 MHz
    dds.exec_at_trg()

    # Ramp the amplitude of the carrier
    dds[0].amplitude_slope(-39 * units.percent / period_s) # 1/s
    dds.exec_at_trg()

    # Stop amplitude ramp
    dds[0].amplitude_slope(0)
    dds[0].amp(1 * units.percent)
    dds.exec_at_trg()

    # Write the list of commands to the card
    dds.write_to_card()

    # Start command including enable of trigger engine
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_FORCETRIGGER)

    input("Press Enter to Exit")