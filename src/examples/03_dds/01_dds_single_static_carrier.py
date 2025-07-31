"""  
Spectrum Instrumentation GmbH (c) 2024

1_dds_single_static_carrier.py

Single static carrier - This example shows the DDS functionality with 1 carrier with a fixed frequency and fixed amplitude

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

    # Setup the channels
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    channels[0].enable(True)
    channels[0].output_load(50 * units.ohm)
    channels[0].amp(0.5 * units.V)
    card.write_setup() # IMPORTANT! this turns on the card's system clock signals, that are required for DDS to work
    
    # Setup DDS functionality
    dds = spcm.DDS(card, channels=channels)
    dds.reset()

    # Start the test
    # dds[0].amp(-20 * units.dBm)
    dds[0].amp(100 * units.mV)
    dds[0].freq(10 * units.MHz)
    dds[0].phase(20 * units.degrees)
    # Read back the exact frequency
    freq = dds[0].get_freq(return_unit=units.MHz)
    amp = dds[0].get_amp(return_unit=units.dBm)
    phase = dds[0].get_phase(return_unit=units.rad)
    print(f"Generated signal frequency: {freq} and amplitude: {amp} and phase: {phase}")
    
    dds.exec_at_trg()
    dds.write_to_card()

    # Start command including enable of trigger engine
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_FORCETRIGGER)

    input("Press Enter to Exit")
