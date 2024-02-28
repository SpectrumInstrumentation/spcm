"""  
Spectrum Instrumentation GmbH (c) 2024

4_dds_ramp_multiple_carriers.py

Ramp multiple carriers - Ramping the frequency of 20 carriers from a one setting to another

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
    num_freq = dds.num_cores()
    # 20 Carriers from 5 to 15 MHz
    first_init_freq_Hz  = 5.0e6 #   5   MHz
    delta_init_freq_Hz  = 5.0e5 # 500   kHz
    # 20 Carriers from 8 to 12 MHz
    first_final_freq_Hz = 8.0e6 #   9 MHz
    delta_final_freq_Hz = 2.0e5 # 100 kHz

    # STEP 0 - Initialize frequencies
    dds.trg_timer(2.0)
    dds.trg_src(spcm.SPCM_DDS_TRG_SRC_TIMER)
    for i in range(num_freq):
        dds.amp(i, 0.45 / num_freq)
        dds.freq(i, first_init_freq_Hz + i * delta_init_freq_Hz)
    dds.exec_at_trg()
    dds.write_to_card()

    # STEP 1 - Start the ramp
    period_s = 5.0 # seconds
    dds.trg_timer(period_s) # after 2.0 s stop the ramp
    for i in range(num_freq):
        dds.frequency_slope(i, (first_final_freq_Hz - first_init_freq_Hz + i * (delta_final_freq_Hz  - delta_init_freq_Hz)) / period_s) # Hz/s
    dds.exec_at_trg()
    
    # STEP 2 - Stop the ramp
    for i in range(num_freq):
        dds.frequency_slope(i, 0) # Hz/s
        dds.freq(i, first_final_freq_Hz + i * delta_final_freq_Hz)
    dds.exec_at_trg()
    dds.write_to_card()

    # Start command including enable of trigger engine
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_FORCETRIGGER)

    input("Press Enter to Exit")
