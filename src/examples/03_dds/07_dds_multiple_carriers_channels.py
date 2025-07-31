"""  
Spectrum Instrumentation GmbH (c) 2024

7_dds_multiple_carriers_channels.py

Multiple static carriers on multiple channels - This example shows the DDS functionality with 20 carriers
with individual but fixed frequencies divided over 4 channels.

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
    channels = spcm.Channels(card)
    channels.enable(True)
    channels.output_load(50 * units.ohm)
    channels.amp(1 * units.V)
    card.write_setup() # IMPORTANT! this turns on the card's system clock signals, that are required for DDS to work
    
    # Setup DDS
    dds = spcm.DDS(card, channels=channels)
    dds.reset()
   
    # Switch groups of cores to other channels
    if len(channels) == 2:
        # pass
        dds.cores_on_channel(1, 
                             spcm.SPCM_DDS_CORE8,  spcm.SPCM_DDS_CORE9,  spcm.SPCM_DDS_CORE10, spcm.SPCM_DDS_CORE11, # Flex core block 8 - 11
                             spcm.SPCM_DDS_CORE20) # Fixed core 20
    elif len(channels) == 4:
        dds.cores_on_channel(1, 
                             spcm.SPCM_DDS_CORE8,  spcm.SPCM_DDS_CORE9,  spcm.SPCM_DDS_CORE10, spcm.SPCM_DDS_CORE11, # Flex core block 8 - 11
                             spcm.SPCM_DDS_CORE20) # Fixed core 20
        dds.cores_on_channel(2,
                             spcm.SPCM_DDS_CORE12, spcm.SPCM_DDS_CORE13, spcm.SPCM_DDS_CORE14, spcm.SPCM_DDS_CORE15, # Flex core block 12 - 15
                             spcm.SPCM_DDS_CORE21) # Fixed core 21
        dds.cores_on_channel(3,
                             spcm.SPCM_DDS_CORE16, spcm.SPCM_DDS_CORE17, spcm.SPCM_DDS_CORE18, spcm.SPCM_DDS_CORE19, # Flex core block 16 - 19
                             spcm.SPCM_DDS_CORE22) # Fixed core 22

    # Start the test
    num_freq     = len(dds)
    # 20 Carriers from 5 to 15 MHz
    start_freq_Hz = 5 * units.MHz
    delta_freq_Hz = 500 * units.kHz
    for core in dds:
        amp = 40 * units.percent/num_freq
        core.amp(amp)
        core.freq(start_freq_Hz + int(core) * delta_freq_Hz)
        print("{} - Frequency: {} - Amplitude: {}".format(core, core.get_freq(return_unit=units.MHz), core.get_amp(return_unit=units.dBm)))
    dds.exec_at_trg()
    dds.write_to_card()

    # Start command including enable of trigger engine
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_FORCETRIGGER)

    input("Press Enter to Exit")
