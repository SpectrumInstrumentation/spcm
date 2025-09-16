"""  
Spectrum Instrumentation GmbH (c) 2024

4_dds_ramp_multiple_carriers.py

Ramp multiple carriers - Ramping the frequency of 20 carriers from a one setting to another

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
    channels.amp(500 * units.mV)
    card.write_setup() # IMPORTANT! this turns on the card's system clock signals, that are required for DDS to work
    
    # Setup DDS
    dds = spcm.DDS(card, channels=channels)
    dds.reset()

    # Start the DDS test
    num_cores = len(dds)
    # 5 to 15 MHz
    first_init_freq_Hz  = 5.0 * units.MHz
    delta_init_freq_Hz  = 500 * units.kHz
    # 8 to 12 MHz
    first_final_freq_Hz = 8.0 * units.MHz
    delta_final_freq_Hz = 200 * units.kHz

    # STEP 0 - Initialize frequencies
    dds.trg_timer(2.0 * units.s)
    dds.trg_src(spcm.SPCM_DDS_TRG_SRC_TIMER)
    for core in dds:
        core.amp(45 * units.percent / num_cores)
        core.freq(first_init_freq_Hz + int(core) * delta_init_freq_Hz)
    dds.exec_at_trg()
    dds.write_to_card()

    # STEP 1 - Start the ramp
    period_s = 5.0 * units.s
    dds.trg_timer(period_s)
    for core in dds:
        core.frequency_slope((first_final_freq_Hz - first_init_freq_Hz + int(core) * (delta_final_freq_Hz  - delta_init_freq_Hz)) / period_s)
    dds.exec_at_trg()
    
    # STEP 2 - Stop the ramp
    for core in dds:
        core.frequency_slope(0) # Hz/s
        core.freq(first_final_freq_Hz + int(core) * delta_final_freq_Hz)
    dds.exec_at_trg()
    dds.write_to_card()

    # Start command including enable of trigger engine
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_FORCETRIGGER)

    input("Press Enter to Exit")
