"""
Spectrum Instrumentation GmbH (c)

6_acq_fifo_multi_ts_poll.py

Shows an example for FIFO mode with timestamp polling

Example for analog recording cards (digitizers) for the the M2p, M4i, M4x and M5i card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
import numpy as np

card : spcm.Card

# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(card_type=spcm.SPCM_TYPE_AI) as card:            # if you want to open the first card of a specific type

    # do a simple standard setup
    card.card_mode(spcm.SPC_REC_FIFO_MULTI) # multiple recording FIFO mode
    card.timeout(5000)

    # setup clock engine
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)
    sample_rate = clock.sample_rate(20e6)

    # setup trigger engine
    trigger = spcm.Trigger(card)
    trigger.ext0_mode(spcm.SPC_TM_POS)   # set trigger mode
    trigger.or_mask(spcm.SPC_TMASK_EXT0) # trigger set to external
    trigger.ext0_coupling(spcm.COUPLING_DC)  # trigger coupling
    trigger.ext0_level0(1500)            # trigger level of 1.5 Volt

    # setup channels
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    amplitude_mV = 1000
    channels.amp(amplitude_mV)
    channels.termination(1)
    max_sample_value = card.max_sample_value()

    # settings for the FIFO mode buffer handling
    total_samples = spcm.KIBI(96) # set this to zero to record forever
    num_samples = spcm.KIBI(48)
    notify_samples = spcm.KIBI(12)
    num_timestamps = spcm.KIBI(8)
    
    # setup data transfer buffer
    num_samples_in_segment = 4096
    num_segments = num_samples // num_samples_in_segment
    multiple_recording = spcm.Multi(card)
    multiple_recording.loops(0)
    multiple_recording.memory_size(num_samples)
    multiple_recording.allocate_buffer(segment_samples=num_samples_in_segment, num_segments=num_segments)
    multiple_recording.to_transfer_samples(total_samples)
    multiple_recording.notify_samples(notify_samples)
    multiple_recording.post_trigger(num_samples_in_segment - 128)

    # setup timestamps
    ts = spcm.TimeStamp(card)
    ts.mode(spcm.SPC_TSMODE_STARTRESET, spcm.SPC_TSCNT_INTERNAL)
    ts.allocate_buffer(num_timestamps)

    print("!!! Using external trigger - please connect a signal to the trigger input !!!")

    # Create second buffer
    ts.start_buffer_transfer(spcm.M2CMD_EXTRA_POLL)

    # start everything
    multiple_recording.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER)

    segment_cnt = 0
    print("Recording... press Ctrl+C to stop")
    try:
        for data_block in multiple_recording:

            print("")
            ts_data_range = ts.poll()
            segment = 0
            for ts_block in ts_data_range:
                timestampVal2 = ts_block[0] / sample_rate   # lower 8 bytes

                # write timestamp value to file
                print("Segment[{}]: Time: {:0.2f} s, Minimum: {:0.3f} mV, Maximum: {:0.3f} mV".format(segment_cnt, timestampVal2, np.min(data_block[:,segment,:])/max_sample_value*amplitude_mV, np.max(data_block[:,segment,:])/max_sample_value*amplitude_mV))
                segment += 1
                segment_cnt += 1
                if segment >= data_block.shape[1]:
                    break
    except KeyboardInterrupt:
        print("Recording stopped by user...")

    print("Finished...")

