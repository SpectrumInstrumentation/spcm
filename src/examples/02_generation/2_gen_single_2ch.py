"""
Spectrum Instrumentation GmbH (c)

2_gen_single_2ch.py

Shows a simple standard mode example using only the few necessary commands.
- There will be a saw-tooth signal generated on channel 0 and the inverse on channel 1.
- This signal will have an amplitude of 1 V.

Example for analog replay cards (AWG) for the the M2p, M4i, M4x and M5i card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
from spcm import units

import numpy as np

card : spcm.Card

# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(card_type=spcm.SPCM_TYPE_AO) as card:          # if you want to open the first card of a specific type
    
    # setup card
    card.card_mode(spcm.SPC_REP_STD_CONTINUOUS)
    card.loops(0) # 0 = loop endless; >0 = n times

    # enable the first channel and setup output amplitude
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0 | spcm.CHANNEL1)
    channels.enable(True)
    channels.output_load(units.highZ)
    channels.amp(1 * units.V)

    # setup the clock
    clock = spcm.Clock(card)
    clock.sample_rate(10 * units.percent) # 10% of the maximum sample rate
    clock.clock_output(False)

    num_samples = 1 * units.MiS # samples per channel

    # setup the trigger mode
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)

    # setup data transfer
    data_transfer = spcm.DataTransfer(card)
    if data_transfer.bytes_per_sample != 2: raise spcm.SpcmException(text="Non 16-bit DA not supported")
    data_transfer.memory_size(num_samples) # size of memory on the card
    data_transfer.allocate_buffer(num_samples) # size of buffer in pc RAM

    # generate output data (or alternatively load data from file)
    samples = num_samples.to_base_units().magnitude
    # simple ramp for analog output cards
    data_transfer.buffer[channels[0], :] = np.arange(-samples//2, samples//2).astype(np.int16) # saw-tooth signal
    data_transfer.buffer[channels[1], :] = - np.arange(-samples//2, samples//2).astype(np.int16) # inverted saw-tooth signal

    data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA, spcm.M2CMD_DATA_WAITDMA) # Wait until the writing to buffer has been done

    # We'll start and wait until the card has finished or until a timeout occurs
    card.timeout(10 * units.s) # 10 s; 0 = disable timeout functionality
    print("Starting the card and waiting for ready interrupt\n(continuous and single restart will have timeout)")
    try:
        card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_WAITREADY)
    except spcm.SpcmTimeout as timeout:
        print("-> The 10 seconds timeout have passed and the card is stopped")

    # Without the above "spcm.M2CMD_CARD_WAITREADY" flag you can do things here in parallel
    # and later stop the replaying with "card.stop()"
