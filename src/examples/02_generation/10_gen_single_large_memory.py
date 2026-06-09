"""
Spectrum Instrumentation GmbH (c)

10_gen_single_large_memory.py

Shows how to fill a large on-card memory (1 GiSample) with waveform data using
chunked DMA transfers, then replay it continuously.

- A sine wave at 1 MHz is generated on channel 0 with an amplitude of 1 V.
- The card memory (8 GiS) is larger than the PC-side buffer (16 MiS), so the
  data is transferred in notification-based chunks of 16 MiS each.
- After the full memory has been written, the card is started and replays the
  waveform endlessly in standard continuous mode.
- A software trigger is used and a 100 s timeout stops the card if no other
  stop condition occurs.

Example for analog replay cards (AWG) for the M2p, M4i, M4x and M5i card-families.

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
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    channels.enable(True)
    channels.output_load(units.highZ)
    channels.amp(1 * units.V)

    # setup the clock
    clock = spcm.Clock(card)
    sample_rate = clock.sample_rate(max=True) # 10% of the maximum sample rate
    clock.clock_output(False)

    mem_samples    =   8 * units.GiS # samples per channel
    notify_samples =  16 * units.MiS # chunk size for transfer
    RAM_samples    =  notify_samples # size of buffer in pc RAM

    # setup the trigger mode
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)

    # setup data transfer
    data_transfer = spcm.DataTransfer(card)
    if data_transfer.bytes_per_sample != 2: raise spcm.SpcmException(text="Non 16-bit DA not supported")
    data_transfer.memory_size(mem_samples) # size of memory on the card^
    data_transfer.notify_samples(notify_samples) # size of chunk for transfer
    data_transfer.allocate_buffer(RAM_samples) # size of buffer in pc RAM
    data_transfer.to_transfer_samples(mem_samples) # total number of samples to transfer

    # generate output data (or alternatively load data from file)
    num_RAM_samples = RAM_samples.to_base_units().magnitude
    num_notify_samples = notify_samples.to_base_units().magnitude

    data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA) # Wait until the writing to buffer has been done

    # Do the transfer in chunks of "notify_samples" and wait for the card to be ready after each chunk
    block_num = 0
    data_range = np.arange(num_notify_samples, dtype=np.float64)
    frequency = (1 * units.MHz).to_base_units().magnitude
    for data_block in data_transfer:
        sin_phase = (2 * np.pi * frequency / sample_rate * (data_range + block_num * num_notify_samples))
        data_block[:] = np.sin(sin_phase) * (2**15 - 1) # sine signal; full scale amplitude for 16-bit DA
        # All the data is pre-calculated and the loading is done here
        print(f"Transferred {block_num*notify_samples} to the card", end="\r")
        block_num += 1

    # We'll start and wait until the card has finished or until a timeout occurs
    card.timeout(100 * units.s) # 100 s; 0 = disable timeout functionality
    print("Starting the card and waiting for ready interrupt\n(continuous and single restart will have timeout)")
    try:
        card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_WAITREADY)
    except spcm.SpcmTimeout as timeout:
        print("-> The 100 seconds timeout have passed and the card is stopped")

    # Without the above "spcm.M2CMD_CARD_WAITREADY" flag you can do things here in parallel
    # and later stop the replaying with "card.stop()"
