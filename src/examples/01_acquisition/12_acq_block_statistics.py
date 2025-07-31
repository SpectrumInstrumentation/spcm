"""
Spectrum Instrumentation GmbH (c)

12_acq_block_statistics.py

Shows a simple Block averaging example using only the few necessary commands
- connect a function generator that generates a sine wave with 10-100 kHz frequency and 200 mV amplitude to channel 0
- triggering is done with a channel trigger on channel 0

Example for analog recording cards (digitizers) for the the M2p, M4i and M4x card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
from spcm import units


card : spcm.Card

# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(card_type=spcm.SPCM_TYPE_AI) as card:            # if you want to open the first card of a specific type
    
    # setup card mode
    card.card_mode(spcm.SPC_REC_STD_SEGSTATS) # block averaging mode
    card.timeout(5 * units.s)
    
    # Trigger settings
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_NONE)

    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)  # Internal clock
    sampling_rate = clock.sample_rate(max=True, return_unit=units.Hz) # Adjusted sample rate

    # Enable and configure Channel 0
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0 | spcm.CHANNEL1)
    channels.amp(1000 * units.mV)  
    channels.offset(0)
    channels.coupling(spcm.COUPLING_DC)  # DC coupling

    trigger.ch_and_mask0(spcm.SPC_TMASK0_CH0)
    trigger.ch_mode(channels[0], spcm.SPC_TM_POS)
    trigger.ch_level0(channels[0], 200 * units.mV, return_unit=units.mV)

    num_segments = 4
    samples_per_segment = 16 * units.KiS
    samples_per_segment_magnitude = samples_per_segment.to_base_units().magnitude
    num_samples = samples_per_segment * num_segments
    post_trigger = samples_per_segment // 2

    # Block Statistics Setup and Data Transfer
    block_statistics = spcm.BlockStatistics(card)
    block_statistics.memory_size(num_samples)  # Define memory segment
    block_statistics.allocate_buffer(samples_per_segment, num_segments)
    block_statistics.post_trigger(post_trigger)

    # setup timestamps (only need to turn them on, the timestamps are automatically saved in the block statistics data)
    ts = spcm.TimeStamp(card)
    ts.mode(spcm.SPC_TSMODE_STARTRESET, spcm.SPC_TSCNT_INTERNAL)

    # Start data acquisition
    block_statistics.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_DATA_WAITDMA)

    print("Finished acquiring...")

    # wait until the transfer has finished
    try:
        channel = 0
        # Retrieve and plot the acquired data
        for i in range(num_segments):
            print(f"---\nSegment {i}")
            for channel in channels:
                print(f"-- {channel}")
                print("   Average:          {:4.3f~P}".format(channel.convert_data(block_statistics.buffer[i]['average'][channel]/samples_per_segment_magnitude), return_unit=units.V))
                print("   Minimum:          {:4.3f~P}".format(channel.convert_data(block_statistics.buffer[i]['minimum'][channel]), return_unit=units.V))
                print("   Maximum:          {:4.3f~P}".format(channel.convert_data(block_statistics.buffer[i]['maximum'][channel]), return_unit=units.V))
                print("   Minimum Position: {:4.3f~P}".format((block_statistics.buffer[i]['minimum_position'][channel]/sampling_rate).to(units.us)))
                print("   Maximum Position: {:4.3f~P}".format((block_statistics.buffer[i]['maximum_position'][channel]/sampling_rate).to(units.us)))
                print("   Timestamp:        {:4.3f~P}".format((block_statistics.buffer[i]['timestamp'][channel]/sampling_rate).to(units.us)))
    except spcm.SpcmTimeout as timeout:
        print("Timeout...")


