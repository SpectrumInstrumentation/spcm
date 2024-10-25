"""
Spectrum Instrumentation GmbH (c)

3_gen_multi.py

Shows a simple standard mode multiple replay example.
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
with spcm.Card(card_type=spcm.SPCM_TYPE_AO, verbose=True) as card:          # if you want to open the first card of a specific type
    
    # setup card
    card.card_mode(spcm.SPC_REP_STD_MULTI)
    card.loops(0) # 0 = loop endless; >0 = n times
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

    num_segments = 4
    num_samples = 512 * units.KiS
    num_samples_per_segment = num_samples // num_segments

    # setup the trigger mode
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_EXT0)
    trigger.ext0_mode(spcm.SPC_TM_POS)
    trigger.ext0_coupling(spcm.COUPLING_DC)
    trigger.ext0_level0(1 * units.V)

    print("External trigger required - please connect a trigger signal on the order of 1-10 Hz to the trigger input EXT0!")

    multiple_replay = spcm.Multi(card)
    if multiple_replay.bytes_per_sample != 2: raise spcm.SpcmException(text="Non 16-bit DA not supported")

    multiple_replay.memory_size(num_samples)
    multiple_replay.allocate_buffer(num_samples_per_segment, num_segments)

    num_samples_magnitude = num_samples.to_base_units().magnitude
    num_samples_per_segment_magnitude = num_samples_per_segment.to_base_units().magnitude

    time_range = multiple_replay.time_data(num_samples_per_segment)
    frequency = 100 * units.kHz

    # simple ramp for analog output cards
    multiple_replay.buffer[0, :, 0] = np.arange(-num_samples_per_segment_magnitude//2, num_samples_per_segment_magnitude//2).astype(np.int16) # saw-tooth signal
    multiple_replay.buffer[1, :, 0] = +(max_value-1)*np.ones((num_samples_per_segment_magnitude,)).astype(np.int16) # maximum
    multiple_replay.buffer[2, :, 0] = -(max_value-1)*np.ones((num_samples_per_segment_magnitude,)).astype(np.int16) # minimum
    multiple_replay.buffer[3, :, 0] = ((max_value-1)*np.sin(2*np.pi*frequency*time_range).magnitude).astype(np.int16) # sine wave

    multiple_replay.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA, spcm.M2CMD_DATA_WAITDMA) # Wait for the writing to buffer being done

    # We'll start
    print("Starting the card and waiting for external trigger")
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_WAITREADY)


