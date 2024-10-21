"""
Spectrum Instrumentation GmbH (c)

6_acq_fifo_multi_ts_poll.py

Shows an example for FIFO multiple-recording mode with timestamp polling. 
- Please connect a trigger signal (3.3 V) to the trigger input EXT0.
- On channel 0, you could for example provide a sinusoidal signal to the input with a frequency of 5 - 20 kHz and amplitude of +/- 0.5 V.

Example for analog recording cards (digitizers) for the the M2p, M4i, M4x and M5i card-families.

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
with spcm.Card(card_type=spcm.SPCM_TYPE_AI) as card:            # if you want to open the first card of a specific type

    # do a simple standard setup
    card.card_mode(spcm.SPC_REC_FIFO_MULTI) # multiple recording FIFO mode
    card.loops(0)
    card.timeout(5 * units.s)

    # setup clock engine
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)
    sample_rate = clock.sample_rate(20 * units.MHz, return_unit=units.MHz)

    # setup trigger engine
    trigger = spcm.Trigger(card)
    trigger.ext0_mode(spcm.SPC_TM_POS)   # set trigger mode
    trigger.or_mask(spcm.SPC_TMASK_EXT0) # trigger set to external
    trigger.ext0_coupling(spcm.COUPLING_DC)  # trigger coupling
    trigger.ext0_level0(1.5 * units.V)

    # setup channels
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    channels.amp(1 * units.V)
    channels.termination(1)

    # settings for the FIFO mode buffer handling
    total_samples = 96 * units.KiS # set this to zero to record forever
    num_samples = 48 * units.KiS
    notify_samples = 12 * units.KiS
    num_timestamps = spcm.KIBI(8)
    
    # setup data transfer buffer
    num_samples_in_segment = 4 * units.KiS
    num_segments = num_samples // num_samples_in_segment
    multiple_recording = spcm.Multi(card)
    multiple_recording.memory_size(num_samples)
    multiple_recording.allocate_buffer(segment_samples=num_samples_in_segment, num_segments=num_segments)
    multiple_recording.to_transfer_samples(total_samples)
    multiple_recording.notify_samples(notify_samples)
    multiple_recording.post_trigger(num_samples_in_segment - 128 * units.S)

    # setup timestamps
    ts = spcm.TimeStamp(card)
    ts.mode(spcm.SPC_TSMODE_STARTRESET, spcm.SPC_TSCNT_INTERNAL)
    ts.allocate_buffer(num_timestamps)

    print("External trigger required - please connect a trigger signal on the order of 1-10 Hz to the trigger input EXT0!")

    # Create second buffer
    ts.start_buffer_transfer(spcm.M2CMD_EXTRA_POLL)

    # start everything
    multiple_recording.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER)

    segment_cnt = 0
    print("Recording... press Ctrl+C to stop")
    try:
        for data_block in multiple_recording:
            segment = 0
            while True:
                ts_data_range = ts.poll()
                for ts_block in ts_data_range:
                    timestampVal2 = (ts_block[0] / sample_rate).to_base_units()   # lower 8 bytes

                    # write timestamp value to file
                    unit_data_block = channels[0].convert_data(data_block[segment,:,:], units.V) # index definition: [segment, sample, channel] !
                    minimum = np.min(unit_data_block)
                    maximum = np.max(unit_data_block)
                    print(f"Segment[{segment_cnt}]: Time: {timestampVal2}, Minimum: {minimum}, Maximum: {maximum}")
                    segment += 1
                    segment_cnt += 1
                    if segment >= data_block.shape[0]:
                        break
                if segment >= data_block.shape[0]:
                    break
    except KeyboardInterrupt:
        print("Recording stopped by user...")

    print("Finished...")

