"""  
Spectrum Instrumentation GmbH (c) 2024

3_dds_ramp_single_carrier.py

Ramp single carrier - Use the ramping functionality to do a long and slow frequency as well as amplitude ramp

Example for analog replay cards (AWG) for the the M4i and M4x card-families with installed DDS option.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm

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
    channels.amp(1000) # 1000 mV
    card.write_setup()
    
    # Setup DDS
    dds = spcm.DDS(card)
    dds.reset()

    # Start the DDS test
    # Timer changes every 2.0 seconds
    period_s = 2.0 # seconds
    dds.trg_src(spcm.SPCM_DDS_TRG_SRC_TIMER)
    dds.trg_timer(period_s)
    # For slow ramps only change the value every 1000 steps
    dds.freq_ramp_stepsize(1000)
    dds.amp_ramp_stepsize(1000)

    # Create one carrier and keep on for 2 seconds
    dds.amp(0, 0.4)
    dds.freq(0, 5e6) # 5 MHz
    dds.exec_at_trg()

    # Ramp the frequency of the carrier
    dds.frequency_slope(0, 10e6 / period_s) # 5 MHz/s
    dds.exec_at_trg()

    # Stop frequency ramp
    dds.frequency_slope(0, 0)
    dds.freq(0, 15e6) # 15 MHz
    dds.exec_at_trg()

    # Ramp the amplitude of the carrier
    dds.amplitude_slope(0, -0.39 / period_s) # 1/s
    dds.exec_at_trg()

    # Stop amplitude ramp
    dds.amplitude_slope(0, 0)
    dds.amp(0, 0.01)
    dds.exec_at_trg()

    # Write the list of commands to the card
    dds.write_to_card()

    # Start command including enable of trigger engine
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_FORCETRIGGER)

    input("Press Enter to Exit")