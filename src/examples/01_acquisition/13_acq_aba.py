"""
Spectrum Instrumentation GmbH (c)

13_acq_aba.py

Shows a simple Standard multiple recording mode example using only the few necessary commands
- connect a function generator that generates a sine wave around 20 kHz frequency and 200 mV RMS amplitude to channel 0
- triggering is done with an external trigger at 1 kHz

Example for analog recording cards (digitizers) for the the M2p, M4i and M4x card-families.

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

    # do a simple standard setup
    card.card_mode(spcm.SPC_REC_STD_ABA) # multiple recording FIFO mode
    card.loops(0)
    card.timeout(5 * units.s)

    # setup clock engine
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)
    sample_rate = clock.sample_rate(20 * units.MHz, return_unit=units.MHz)

    # setup trigger engine
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_EXT0) # trigger set to external
    trigger.ext0_mode(spcm.SPC_TM_POS)   # set trigger mode
    trigger.ext0_coupling(spcm.COUPLING_DC)  # trigger coupling
    trigger.ext0_level0(0.2 * units.V)

    # setup channels
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    channels.amp(1 * units.V)
    channels.termination(1)

    # settings for the FIFO mode buffer handling
    num_samples = 8 * units.KiS
    notify_samples = 12 * units.KiS
    num_aba_samples = 2 * units.KiS
    pre_trigger = 32 * units.S
    pre_trigger_magnitude = pre_trigger.to_base_units().magnitude
    
    # setup data transfer buffer
    num_samples_in_segment = 2 * units.KiS
    num_segments = num_samples // num_samples_in_segment
    data_transfer = spcm.Multi(card)
    data_transfer.memory_size(num_samples)
    data_transfer.allocate_buffer(segment_samples=num_samples_in_segment, num_segments=num_segments)
    data_transfer.post_trigger(num_samples_in_segment - pre_trigger)
    num_timestamps = num_segments + 1

    # setup ABA mode
    aba = spcm.ABA(card)
    divider = aba.divider(48)
    aba.allocate_buffer(num_aba_samples)

    # setup TimeStamps
    ts = spcm.TimeStamp(card)
    ts.mode(spcm.SPC_TSMODE_STARTRESET, spcm.SPC_TSCNT_INTERNAL, spcm.SPC_TSFEAT_STORE1STABA)
    ts.allocate_buffer(num_timestamps)


    print("External trigger required - please connect a trigger signal on the order of 1-10 Hz to the trigger input EXT0!")


    # wait until the transfer has finished
    try:
        
        # Create Timestamp and ABA buffer and start the DMA transfers together
        ts.start_buffer_transfer()
        aba.start_buffer_transfer(spcm.M2CMD_EXTRA_STARTDMA)

        # Start the card and wait until all the data has been recorder
        card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_WAITREADY)
        
        # Create and start multiple recording buffer
        data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA, spcm.M2CMD_DATA_WAITDMA, spcm.M2CMD_EXTRA_WAITDMA)

        aba_time_offset = data_transfer.convert_time(ts.buffer[0, 0]) # timestamps are recorded on the time base of the fast recording
                                                                         # hence the conversion of the multiple_recording object is needed.
        aba_time_data = aba.time_data() + aba_time_offset

        fig = plt.figure()
        ax = plt.gca()
        # Plot the "slow" A data
        for channel in channels:
            chan_data = channel.convert_data(aba.buffer[channel, :] ) # index definition: [segment, sample, channel]
            plt.plot(aba_time_data, chan_data, '.-', label="(A)")
        
        data = data_transfer.buffer
        time_data = data_transfer.time_data()
        # Plot the "fast" B data
        for segment in range(data.shape[0]):
            time_offset = data_transfer.convert_time(ts.buffer[segment + 1, 0])
            ax.axvline(time_offset, color='k', linestyle='--', label='Trigger {}'.format(segment))
            for channel in channels:
                chan_data = channel.convert_data(data[segment, :, channel]) # index definition: [segment, sample, channel]
                plt.plot(time_offset+time_data, chan_data, '.-', label="(B) {}".format(channel))
        ax.yaxis.set_units(units.V)
        ax.xaxis.set_units(units.us)
        ax.legend()
        plt.show()

    except spcm.SpcmTimeout as timeout:
        print("Timeout...")
        card.status()


