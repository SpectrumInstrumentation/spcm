"""  
Spectrum Instrumentation GmbH (c) 2024

6_dds_time_continuous_changes.py

Continuous changes - use one carrier to jump between different frequencies that are send through the FIFO

Example for analog replay cards (AWG) for the the M4i and M4x card-families with installed DDS option.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
from spcm import units


import psutil
import os

# Set the highest process priority to the Python process, to enable highest possible command streaming
p = psutil.Process(os.getpid())
p.nice(psutil.REALTIME_PRIORITY_CLASS)

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
    channels.amp(500 * units.mV)
    card.write_setup() # IMPORTANT! this turns on the card's system clock signals, that are required for DDS to work
    
    # Setup DDS
    dds = spcm.DDS(card, channels=channels)
    dds.reset()

    # Start the DDS test
    num_freq      = 100
    start_freq_Hz  = 5.0 * units.MHz
    delta_freq_Hz  = 100 * units.kHz
    
    freq_list = []
    for i in range(num_freq):
        freq_list.append(start_freq_Hz + i*delta_freq_Hz)

    # STEP 0 - Initialize frequencies
    period_s = 1.0 * units.s
    dds.trg_src(spcm.SPCM_DDS_TRG_SRC_TIMER)
    dds.trg_timer(period_s)
    dds[0].amp(10 * units.percent)
    dds[0].freq(freq_list[0])
    dds.exec_at_trg()
    dds.write_to_card()
    
    # STEP 1a - Pre-fill Buffer
    period_s = 1 * units.ms
    dds.trg_timer(period_s)
    fill_maximum = dds.queue_cmd_max()
    fill_number = int(fill_maximum / 4)
    print("Pre-fill buffer")
    column = 0
    for i in range(fill_number):
        freq_Hz = freq_list[column % num_freq]
        dds[0].freq(freq_Hz)
        dds.exec_at_trg()
        column += 1
    dds.write_to_card()

    # Start the card
    print("Card started. Press Ctrl+C to stop.")
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_FORCETRIGGER)

    # STEP 1b - Streaming data points
    fill_count = dds.queue_cmd_count()
    fill_check = fill_maximum - fill_number
    try:
        while True: # infinitely long streaming
            while True:
                fill_count = dds.queue_cmd_count()
                if fill_count < fill_check: break
            print("Adding a block of commands to buffer")
            for i in range(fill_number):
                freq_Hz = freq_list[column % num_freq]
                dds[0].freq(freq_Hz)
                dds.exec_at_trg()
                column += 1
                column %= num_freq
            dds.write_to_card()
            status = dds.status()
            if status & spcm.SPCM_DDS_STAT_QUEUE_UNDERRUN:
                break
            
        print("ERROR: Buffer underrun")
    except KeyboardInterrupt:
        print("Interrupted by the user")
    