"""
Spectrum Instrumentation GmbH (c)

3_acq_multi.py

Shows a simple Standard multiple recording mode example using only the few necessary commands
- connect a function generator that generates a sine wave with 10-100 kHz frequency and 200 mV amplitude to channel 0
- triggering is done with a channel trigger on channel 0

Example for analog recording cards (digitizers) for the the M2p, M4i, M4x and M5i card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
from spcm import units 

import numpy as np
import matplotlib.pyplot as plt


card : spcm.Card

# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(card_type=spcm.SPCM_TYPE_AI, verbose=True) as card:            # if you want to open the first card of a specific type
    
    # setup card mode
    card.card_mode(spcm.SPC_REC_STD_MULTI) # multiple recording mode
    card.timeout(5 * units.s)

    # setup trigger engine
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_NONE)

    # setup clock engine
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)
    clock.sample_rate(max=True)

    # setup channel 0
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    channels.amp(1 * units.V)

    # Channel triggering
    trigger.ch_or_mask0(channels[0].ch_mask())
    trigger.ch_mode(channels[0], spcm.SPC_TM_POS)
    trigger.ch_level0(channels[0], 0 * units.mV, return_unit=units.mV)

    # setup data transfer
    num_samples = 4 * units.KiS
    samples_per_segment = 1 * units.KiS
    multiple_recording = spcm.Multi(card)
    multiple_recording.memory_size(num_samples)
    multiple_recording.allocate_buffer(samples_per_segment)
    multiple_recording.post_trigger(samples_per_segment // 2)
    multiple_recording.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)


    # wait until the transfer has finished
    try:
        card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_DATA_WAITDMA)

        data = multiple_recording.buffer
        time_data = multiple_recording.time_data()

        # this is the point to do anything with the data
        # e.g. calculate minimum and maximum of the acquired data
        fig, ax = plt.subplots(data.shape[0], 1, sharex=True, layout='constrained')
        for segment in range(data.shape[0]):
            print("Segment {}".format(segment))
            for channel in channels:
                chan_data = channel.convert_data(data[segment, :, channel]) # index definition: [segment, sample, channel]
                minimum = np.min(chan_data)
                maximum = np.max(chan_data)
                print(f"\t{channel}")
                print(f"\t\tMinimum: {minimum}")
                print(f"\t\tMaximum: {maximum}")

                ax[segment].plot(time_data, chan_data, '.', label="{}, Seg {}".format(channel, segment))
            ax[segment].set_title(f"Segment {segment}")
            ax[segment].yaxis.set_units(units.V)
            ax[segment].xaxis.set_units(units.us)
            ax[segment].axvline(0, color='k', linestyle='--', label='Trigger')
        # ax.legend()
        plt.show()
    except spcm.SpcmTimeout as timeout:
        print("Timeout...")


