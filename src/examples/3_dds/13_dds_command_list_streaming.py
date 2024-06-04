"""  
Spectrum Instrumentation GmbH (c) 2024

13_dds_command_list_streaming.py

Similar to example 6_dds_time_continuous_changes.py: 
Continuous changes - use one carrier to jump between different frequencies that are send through the FIFO.
This example uses the DDSCommandList class to write commands to the card in a more efficient way using a 
list mechanism that writes blocks of pre-calculated data to the card.

Example for analog replay cards (AWG) for the the M4i and M4x card-families with installed DDS option.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
from spcm import units

import psutil
import os
import numpy as np

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
    channels.amp(1 * units.V)
    card.write_setup()
    
    # Setup DDS
    dds = spcm.DDSCommandList(card)
    # Please note that units aren't supported in the DDSCommandQueue class, to improve performance
    dds.reset()

    dds.data_transfer_mode(spcm.SPCM_DDS_DTM_DMA)

    # Start the card and enable trigger, but don't send a force trigger yet
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER)
    print("Card started")

    # Start the DDS test
    num_freq      =  20
    start_freq_Hz =   5.0 * 1e6
    delta_freq_Hz = 500.0 * 1e3

    # STEP 0 - Initialize frequencies
    period_s = 1.0
    dds.trg_src(spcm.SPCM_DDS_TRG_SRC_TIMER)
    dds.trg_timer(period_s)
    dds.amp(0, 0.1)
    dds.freq(0, start_freq_Hz)
    dds.exec_at_trg()
    dds.write_to_card()

    print("Calculate frequencies and add to queue")
    period_s = 1000e-9
    dds.trg_timer(period_s)
    dds.write_to_card()

    # Load frequencies to the queue
    freq_list = np.linspace(start_freq_Hz, start_freq_Hz + num_freq*delta_freq_Hz, num_freq)
    dds.load({spcm.SPC_DDS_CORE0_FREQ: freq_list}, exec_mode=spcm.SPCM_DDS_CMD_EXEC_AT_TRG, repeat=0)
    # Preload frequencies to the card
    dds.write()
    
    # Start streaming
    dds.mode = dds.WRITE_MODE.WAIT_IF_FULL

    # Start the card
    card.cmd(spcm.M2CMD_CARD_FORCETRIGGER)
    print("Card triggered")
    print("Streaming... stop by pressing Ctrl+C")

    try:
        while True: # infinitely long streaming
            dds.write()
            if dds.status() & spcm.SPCM_DDS_STAT_QUEUE_UNDERRUN:
                break

        print("ERROR: Buffer underrun")
    except KeyboardInterrupt:
        print("Ctrl+C pressed: streaming stopped by user")