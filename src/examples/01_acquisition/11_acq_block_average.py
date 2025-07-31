"""
Spectrum Instrumentation GmbH (c)

11_acq_block_average.py

Shows a simple Block averaging example using only the few necessary commands
- connect a function generator that generates a sine wave with 10-100 kHz frequency and 200 mV amplitude to channel 0
- triggering is done with a channel trigger on channel 0

Example for analog recording cards (digitizers) for the the M2p, M4i, M4x and M5i card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
from spcm import units 

import matplotlib.pyplot as plt


card : spcm.Card

# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(card_type=spcm.SPCM_TYPE_AI) as card:            # if you want to open the first card of a specific type
    
    # setup card mode
    card.card_mode(spcm.SPC_REC_STD_AVERAGE) # block averaging mode
    card.timeout(5 * units.s)
    
    # Trigger settings
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_NONE)
    trigger = spcm.Trigger(card)

    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)  # Internal clock
    sampling_rate = clock.sample_rate(max=True) # Adjusted sample rate

    # Enable and configure Channel 0
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0 | spcm.CHANNEL1 | spcm.CHANNEL2 | spcm.CHANNEL3) # Enable all channels
    channels.amp(1000 * units.mV)  
    channels.offset(0)
    channels.termination(1)  # HF (50 Ω)
    channels.coupling(spcm.COUPLING_DC)  # DC coupling

    trigger.ch_and_mask0(spcm.SPC_TMASK0_CH0)
    trigger.ch_mode(channels[0], spcm.SPC_TM_POS)
    trigger.ch_level0(channels[0], 0 * units.mV, return_unit=units.mV)

    num_segments = 2
    samples_per_segment = 32 * units.KiS
    num_samples = samples_per_segment * num_segments
    averages = 16
    post_trigger = samples_per_segment - 1024 * units.S 

    # Block Averaging Setup and Data Transfer
    block_average = spcm.BlockAverage(card)
    block_average.averages(averages)  # Set averaging factor
    block_average.memory_size(num_samples)  # Define memory segment
    block_average.allocate_buffer(samples_per_segment, num_segments)
    block_average.post_trigger(post_trigger)
    
    # Start data acquisition
    block_average.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_DATA_WAITDMA)

    print("Finished acquiring...")

    # wait until the transfer has finished
    try:
        fig, ax = plt.subplots(num_segments, 1, sharex=True)

        # Retrieve and plot the acquired data
        time_data_s = block_average.time_data()
        for i in range(num_segments):
            if num_segments > 1:
                cax = ax[i]
            else:
                cax = ax
            cax.set_title(f"Segment {i}")
            for channel in channels:
                channel_data = channel.convert_data(block_average.buffer[i, :, channel], averages=averages, return_unit=units.V)
                cax.plot(time_data_s, channel_data, label=f"{channel}")
            cax.xaxis.set_units(units.us)
            cax.axvline(0, color='k', linestyle='--', label='Trigger')
            cax.legend()
        plt.show()
    except spcm.SpcmTimeout as timeout:
        print("Timeout...")


