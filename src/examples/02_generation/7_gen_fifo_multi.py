"""
Spectrum Instrumentation GmbH (c)

7_gen_fifo_multi.py

Shows a simple fifo multiple replay mode example.
- There are 4 segments.
- Each segment has 512k samples.
- The segments are a saw-tooth signal, a maximum, a minimum and a sine wave.
- The segments are played in a loop.
- The card is triggered by an external trigger on ext0 of 1 Vdc

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
    card.card_mode(spcm.SPC_REP_FIFO_MULTI)
    max_value = card.max_sample_value()

    # Enable all the channels and setup amplitude
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    channels.enable(True)
    channels.output_load(units.highZ)
    channels.amp(1 * units.V)

    # Setup the clock
    clock = spcm.Clock(card)
    clock.sample_rate(10 * units.percent) # 10% of the maximum sample rate
    clock.clock_output(False)

    num_samples_per_segment = 128 * units.KiS
    num_segments_buffer = 16 # the number of segments in the buffer
    num_segments_notify = 4 # the number of segments to notify

    # setup the trigger mode
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_EXT0)
    trigger.ext0_mode(spcm.SPC_TM_POS)
    trigger.ext0_coupling(spcm.COUPLING_DC)
    trigger.ext0_level0(1 * units.V)

    print("External trigger required - please connect a trigger signal on the order of 1-10 Hz to the trigger input EXT0!")

    multiple_replay = spcm.Multi(card)
    if multiple_replay.bytes_per_sample != 2: raise spcm.SpcmException(text="Non 16-bit DA not supported")

    multiple_replay.notify_samples(num_segments_notify * num_samples_per_segment)
    multiple_replay.allocate_buffer(num_segments=num_segments_buffer, segment_samples=num_samples_per_segment)

    num_samples_per_segment_magnitude = num_samples_per_segment.to_base_units().magnitude

    time_range = multiple_replay.time_data(num_samples_per_segment)
    frequency = 100 * units.kHz

    # simple ramp for analog output cards
    data = np.zeros((num_segments_notify, num_samples_per_segment_magnitude), dtype=np.int16)
    data[0, :] = np.arange(-num_samples_per_segment_magnitude//2, num_samples_per_segment_magnitude//2).astype(np.int16) # saw-tooth signal
    data[1, :] = +(max_value-1)*np.ones((num_samples_per_segment_magnitude,)).astype(np.int16) # maximum
    data[2, :] = -(max_value-1)*np.ones((num_samples_per_segment_magnitude,)).astype(np.int16) # minimum
    data[3, :] = ((max_value-1)*np.sin(2*np.pi*frequency*time_range).magnitude).astype(np.int16) # sine wave

    # Initialize the buffer
    multiple_replay.buffer[:4, :, 0] = data
    multiple_replay.buffer[4:8, :, 0] = data
    multiple_replay.buffer[8:12, :, 0] = data

    # Define the buffer and do a first transfer
    multiple_replay.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)

    # Continue loading the buffer
    for data_block in multiple_replay:
        # Check the filling of the buffer
        fill_size = multiple_replay.fill_size_promille()
        print("Filling: {}%".format(fill_size/10), end="\r")
        if fill_size == 1000:
            multiple_replay.flush()
            break
        data_block[:,:,0] = data

    print("... data has been transferred to board memory")

    # We'll start
    print("Starting the card and waiting for external trigger. Press Ctrl+C to stop the card.")
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER)

    try:
        for data_block in multiple_replay:
            print("Filling: {}%".format(multiple_replay.fill_size_promille()/10), end="\r")
            data_block[:,:,0] = data
    except spcm.SpcmException as exception:
        # Probably a buffer underrun has happened, capure the event here
        print(exception)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected, stopping the card")

