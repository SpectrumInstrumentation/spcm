"""  
Spectrum Instrumentation GmbH (c) 2024

1_dds_single_static_carrier.py

Single static carrier - This example shows the DDS functionality with 1 carrier with a fixed frequency and fixed amplitude

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

    # Setup the channels
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    channels.enable(True)
    channels.amp(1000) # 1000 mV
    card.write_setup()
    
    # Setup DDS functionality
    dds = spcm.DDS(card)
    dds.reset()

    # Start the test
    dds[0].amp(0.4)
    dds[0].freq(10e6) # 10 MHz
    # Read back the exact frequency
    freq_Hz = dds[0].get_freq()
    amp = dds[0].get_amp()
    print(f"Generated signal frequency: {freq_Hz} Hz and amplitude: {amp}")
    
    dds.exec_at_trg()
    dds.write_to_card()

    # Start command including enable of trigger engine
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_FORCETRIGGER)

    input("Press Enter to Exit")
