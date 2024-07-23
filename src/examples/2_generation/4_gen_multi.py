"""
Spectrum Instrumentation GmbH (c)

4_gen_multi.py

Shows a simple standard mode multiple replay example.
- There are 4 segments.
- Each segment has 512k samples.
- The segments are a saw-tooth signal, a maximum, a minimum and a sine wave.
- The segments are played in a loop.
- The card is triggered by an external trigger on ext0 of 1 Vdc

Example for analog replay cards (AWG) for the the M2p, M4i and M4x card-families.

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
with spcm.Card(card_type=spcm.SPCM_TYPE_AO) as card:             # if you want to open the first card of a specific type
    
    # setup card
    card.card_mode(spcm.SPC_REP_STD_MULTI)
    max_value = card.max_sample_value()

    # Enable all the channels and setup amplitude
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    channels.enable(True)
    channels.output_load(units.highZ)
    channels.amp(1 * units.V)

    # Setup the clock
    clock = spcm.Clock(card)
    series = card.series()
    # set samplerate to 50 MHz (M4i) or 1 MHz (otherwise), no clock output
    if (series in [spcm.TYP_M4IEXPSERIES, spcm.TYP_M4XEXPSERIES, spcm.TYP_M5IEXPSERIES]):
        clock.sample_rate(50 * units.MHz)
    else:
        clock.sample_rate(1 * units.MHz)
    clock.clock_output(0)

    num_segments = 4
    num_samples = 512 * units.KiS
    num_samples_per_segment = num_samples // num_segments


    # setup the trigger mode
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_EXT0)
    trigger.ext0_mode(spcm.SPC_TM_POS)
    trigger.ext0_coupling(spcm.COUPLING_DC)
    trigger.ext0_level0(1 * units.V)

    multiple_replay = spcm.Multi(card)
    if multiple_replay.bytes_per_sample != 2: raise spcm.SpcmException(text="Non 16-bit DA not supported")

    multiple_replay.memory_size(num_samples)
    multiple_replay.allocate_buffer(num_samples_per_segment, num_segments)
    multiple_replay.loops(0) # loop continuously

    num_samples_magnitude = num_samples.to_base_units().magnitude
    num_samples_per_segment_magnitude = num_samples_per_segment.to_base_units().magnitude

    time_range = multiple_replay.time_data(num_samples_per_segment)
    frequency = 100 * units.kHz

    # simple ramp for analog output cards
    multiple_replay.buffer[0, :, 0] = np.arange(-num_samples_per_segment_magnitude//2, num_samples_per_segment_magnitude//2).astype(np.int16) # saw-tooth signal
    multiple_replay.buffer[1, :, 0] = +(max_value-1)*np.ones((num_samples_per_segment_magnitude,)).astype(np.int16) # maximum
    multiple_replay.buffer[2, :, 0] = -(max_value-1)*np.ones((num_samples_per_segment_magnitude,)).astype(np.int16) # minimum
    multiple_replay.buffer[3, :, 0] = (max_value-1)*np.sin(2*np.pi*frequency*time_range) # sine wave

    multiple_replay.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA, spcm.M2CMD_DATA_WAITDMA) # Wait for the writing to buffer being done

    # We'll start
    print("Starting the card and waiting for ready interrupt\n(continuous and single restart will have timeout)")
    try:
        card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_WAITREADY)
    except KeyboardInterrupt:
        print("-> Ctrl+C pressed, execution stopped")


