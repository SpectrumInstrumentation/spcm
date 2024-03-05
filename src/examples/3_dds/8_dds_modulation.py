"""  
Spectrum Instrumentation GmbH (c) 2024

8_dds_modulation.py

Modulation - use continuous changes to modulate the frequency of a carrier

Example for analog replay cards (AWG) for the the M4i and M4x card-families with installed DDS option.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
import numpy as np

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
    channels.amp(1000) # 1000 mV
    card.write_setup()
    
    # Setup DDS
    dds = spcm.DDS(card)
    dds.reset()

    # Set the data transfer mode to DMA
    dds.data_transfer_mode(spcm.SPCM_DDS_DTM_DMA)

    # Start the DDS test
    carrier_freq_Hz  =   10e6 # 10 MHz
    modulation_freq_Hz  = 1   #  1 Hz
    modulation_depth_Hz = 1e6 #  1 MHz
    period_s = 20e-3 # 20 ms
    num_samples = int(1/(period_s * modulation_freq_Hz))
    
    sample_range = np.arange(num_samples)*period_s
    freq_list = carrier_freq_Hz + modulation_depth_Hz * np.sin(2*np.pi*sample_range*modulation_freq_Hz)

    # STEP 0 - Initialize frequencies
    dds.trg_src(spcm.SPCM_DDS_TRG_SRC_TIMER)
    dds.trg_timer(1.0)
    core0 = dds[0]
    core0.amp(0.1)
    core0.freq(carrier_freq_Hz)
    dds.exec_at_trg()
    dds.write_to_card()
    
    # STEP 1 - Pre-fill Buffer
    dds.trg_timer(period_s)
    dds.exec_at_trg()
    fill_max = dds.queue_cmd_max()
    counter = 0
    for counter in range(fill_max // 2 - 4):
        freq_Hz = freq_list[counter % num_samples]
        core0.freq(freq_Hz)
        dds.exec_at_trg()
    dds.write_to_card()
    
    # Start the card
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_FORCETRIGGER)
    
    fill_count = dds.queue_cmd_count()
    print("Pre-fill buffer: {}/{}".format(fill_count, fill_max))

    # STEP 2 - Streaming data points
    fill_number = fill_max // 4
    fill_check = fill_max - fill_number
    while True: # infinitely long streaming
        while True:
            fill_count = dds.queue_cmd_count()
            if fill_count < fill_check: break
        print("Adding a block of commands to buffer")
        for i in range(fill_number // 2):
            freq_Hz = freq_list[counter % num_samples]
            core0.freq(freq_Hz)
            dds.exec_at_trg()
            counter += 1
            counter %= num_samples
        dds.write_to_card()
        status = dds.status()
        if status & spcm.SPCM_DDS_STAT_QUEUE_UNDERRUN:
            break

    print("ERROR: Buffer underrun")