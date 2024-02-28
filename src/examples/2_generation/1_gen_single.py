"""
Spectrum Instrumentation GmbH (c)

1_gen_single.py

Shows a simple standard mode example using only the few necessary commands

Example for analog replay cards (AWG) for the the M2p, M4i and M4x card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
import numpy as np

card : spcm.Card

# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(card_type=spcm.SPCM_TYPE_AO) as card:             # if you want to open the first card of a specific type
    
    # setup card
    card.card_mode(spcm.SPC_REP_STD_CONTINUOUS)

    # Enable all the channels and setup amplitude
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    channels.amp(1000) # 1000 mV
    channels.enable(True)

    # Setup the clock
    clock = spcm.Clock(card)
    series = card.series()
    # set samplerate to 50 MHz (M4i) or 1 MHz (otherwise), no clock output
    if (series in [spcm.TYP_M4IEXPSERIES, spcm.TYP_M4XEXPSERIES]):
        sample_rate = clock.sample_rate(spcm.MEGA(50))
    else:
        sample_rate = clock.sample_rate(spcm.MEGA(1))
    clock.clock_output(0)

    num_samples = spcm.MEBI(32)

    # setup the trigger mode
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)

    data_transfer = spcm.DataTransfer(card)
    if data_transfer.bytes_per_sample != 2: raise spcm.SpcmException(text="Non 16-bit DA not supported")

    data_transfer.memory_size(num_samples)
    data_transfer.allocate_buffer(num_samples)
    data_transfer.loops(0) # loop continuously
    # simple ramp for analog output cards
    data_transfer.buffer[:] = np.arange(-num_samples//2, num_samples//2).astype(np.int16)

    data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA, spcm.M2CMD_DATA_WAITDMA) # Wait for the writing to buffer being done

    # We'll start and wait until the card has finished or until a timeout occurs
    card.timeout(10000) # 10 s
    print("Starting the card and waiting for ready interrupt\n(continuous and single restart will have timeout)")
    try:
        card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_WAITREADY)
    except spcm.SpcmTimeout as timeout:
        print("-> The 10 seconds timeout have passed and the card is stopped")


