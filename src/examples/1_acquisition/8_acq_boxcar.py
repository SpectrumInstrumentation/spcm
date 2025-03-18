"""
Spectrum Instrumentation GmbH (c)

8_acq_boxcar.py

Shows a simple Boxcar averaging example using only the few necessary commands
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
with spcm.Card(card_type=spcm.SPCM_TYPE_AI, verbose=True) as card:            # if you want to open the first card of a specific type
    
    # setup card mode
    card.card_mode(spcm.SPC_REC_STD_BOXCAR) # multiple recording mode
    card.timeout(5 * units.s)# Clock Configuration
    
    # Trigger settings
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_NONE)

    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)  # Internal clock
    sampling_rate = clock.sample_rate(max=True) # Adjusted sample rate

    # Enable and configure Channel 0
    channel0, = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    channel0.amp(1000 * units.mV)  
    channel0.offset(0)
    channel0.termination(0)  # High impedance (1 MΩ)
    channel0.coupling(spcm.COUPLING_DC)  # DC coupling

    trigger.ch_and_mask0(spcm.SPC_TMASK0_CH0)
    trigger.ch_mode(channel0, spcm.SPC_TM_POS)
    trigger.ch_level0(channel0, 200 * units.mV, return_unit=units.mV)

    num_samples = 4 * units.KiS
    num_segments = 1
    samples_per_segment = num_samples // num_segments
    averages = 8
    post_trigger = samples_per_segment // 2
    # Boxcar Averaging Setup and Data Transfer
    boxcar = spcm.Boxcar(card)
    boxcar.box_averages(averages)  # Set boxcar averaging factor
    boxcar.memory_size(num_samples)  # Define memory segment
    boxcar.allocate_buffer(samples_per_segment, num_segments)
    boxcar.post_trigger(post_trigger)
    
    # Start data acquisition
    boxcar.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_DATA_WAITDMA)

    print("Finished acquiring...")

    # wait until the transfer has finished
    try:
        fig, ax = plt.subplots()

        # Retrieve and plot the acquired data
        time_data_s = boxcar.time_data()
        for i in range(num_segments):
            channel_data = channel0.convert_data(boxcar.buffer[i, :, channel0], units.V)
            # Plot the results, only every 256th value to increase visibility of data points
            ax.plot(time_data_s, channel_data, label=f"Segment {i}")
        ax.xaxis.set_units(units.us)
        ax.axvline(0, color='k', linestyle='--', label='Trigger')
        ax.legend()
        plt.show()
    except spcm.SpcmTimeout as timeout:
        print("Timeout...")


