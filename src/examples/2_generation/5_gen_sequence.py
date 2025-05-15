"""
Spectrum Instrumentation GmbH (c)

5_gen_sequence.py

Shows a simple sequence mode example using only the few necessary commands.
- output on channel 0
- 10% of the maximum sampling rate of the card

Example for analog replay cards (AWG) for the the M2p, M4i and M4x card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
from spcm import units

import numpy as np

import msvcrt

USING_EXTERNAL_TRIGGER = False

def kb_hit():
    """
    get the key that was pressed

    Returns
    -------
    int
        the ASCII code of the key that was pressed
    """
    return ord(msvcrt.getch()) if msvcrt.kbhit() else 0


def setup_sequence(sequence : spcm.Sequence):
    print("Calculation of output data")

    # helper values
    full_scale = sequence.card.max_sample_value() - 1
    half_scale = full_scale // 2

    ### Programming the segments
    # we generate the same data on all active channels !

    # sync puls: first half zero, second half -FS
    num_samples_in_segment = 480 #factor * 80
    segment_sync = sequence.add_segment(num_samples_in_segment)
    segment_sync[:, :num_samples_in_segment // 2] = 0
    segment_sync[:, num_samples_in_segment // 2:] = -full_scale

    # sine
    num_samples_in_segment = 768 # factor * 128
    segment_sin1 = sequence.add_segment(num_samples_in_segment)
    segment_sin1[:, :] = half_scale + np.int32(half_scale * np.sin(2.0 * np.pi * (np.arange(num_samples_in_segment)) / num_samples_in_segment) + 0.5)

    # cosine
    segment_sin2 = sequence.add_segment(num_samples_in_segment)
    segment_sin2[:, :] = half_scale + np.int32(half_scale * np.sin(2 * 2.0 * np.pi * (np.arange(num_samples_in_segment)) / num_samples_in_segment) + 0.5)

    # inverted sine
    segment_sin3 = sequence.add_segment(num_samples_in_segment)
    segment_sin3[:, :] = half_scale + np.int32(half_scale * np.sin(3 * 2.0 * np.pi * (np.arange(num_samples_in_segment)) / num_samples_in_segment) + 0.5)

    # inverted cosine
    segment_sin4 = sequence.add_segment(num_samples_in_segment)
    segment_sin4[:, :] = half_scale + np.int32(half_scale * np.sin(4 * 2.0 * np.pi * (np.arange(num_samples_in_segment)) / num_samples_in_segment) + 0.5)

    # DC level
    segment_dc_level = sequence.add_segment(num_samples_in_segment)
    segment_dc_level[:, :] = full_scale // 2
    
    ### Programming the steps
    # The sequence is divided into four parts:
    #  part 0: sync, sin 1
    #  part 1: sync, sin 2
    #  part 2: sync, sin 3
    #  part 3: sync, sin 4
    #  part 4: dc-level (final step)

    # sine
    part0_sync   = sequence.add_step(segment_sync, loops=1)
    part0_sin1   = sequence.add_step(segment_sin1, loops=2)

    # cosine
    part1_sync   = sequence.add_step(segment_sync, loops=1)
    part1_sin2   = sequence.add_step(segment_sin2, loops=2)

    # inverted sine
    part2_sync     = sequence.add_step(segment_sync, loops=1)
    part2_sin3     = sequence.add_step(segment_sin3, loops=2)

    # inverted cosine
    part3_sync     = sequence.add_step(segment_sync, loops=1)
    part3_sin4     = sequence.add_step(segment_sin4, loops=2)

    # final step: DC level
    final_dc       = sequence.add_step(segment_dc_level, loops=1)

    ### Programming the transitions between the different steps

    # Configure which step is executed first
    sequence.entry_step(part0_sync)
    
    # Transitions in the first part of the sequence
    # NOTE: if no external trigger is used, the transition is set to the same step and with a key press
    # the step is changed and a transitions to the next set is set
    part0_sync.set_transition(part0_sin1)
    if USING_EXTERNAL_TRIGGER:
        part0_sin1.set_transition(part1_sync, on_trig=True)
    else:
        part0_sin1.set_transition(part0_sin1)
    
    # The second part of the sequence
    part1_sync.set_transition(part1_sin2)
    if USING_EXTERNAL_TRIGGER:
        part1_sin2.set_transition(part2_sync, on_trig=True)
    else:
        part1_sin2.set_transition(part1_sin2)
    
    # The third part of the sequence
    part2_sync.set_transition(part2_sin3)
    if USING_EXTERNAL_TRIGGER:
        part2_sin3.set_transition(part3_sync, on_trig=True)
    else:
        part2_sin3.set_transition(part2_sin3)
    
    # The fourth part of the sequence
    part3_sync.set_transition(part3_sin4)
    if USING_EXTERNAL_TRIGGER:
        part3_sin4.set_transition(final_dc, on_trig=True)
    else:
        part3_sin4.set_transition(part3_sin4)

    # This is the final step of the sequence
    final_dc.final_step()
    
    # write the segments to the card
    sequence.write_setup()

    print("... sequence setup done")


card : spcm.Card
# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(card_type=spcm.SPCM_TYPE_AO, verbose=False) as card:          # if you want to open the first card of a specific type
    
    # setup card mode
    card.card_mode(spcm.SPC_REP_STD_SEQUENCE)
    
    # set up the channels
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    channels.enable(True)
    channels.output_load(units.highZ) # high impedance
    channels.amp(1 * units.V)
    channels.stop_level(spcm.SPCM_STOPLVL_HOLDLAST)

    # set up the mode
    sequence = spcm.Sequence(card)

    # set up trigger
    trigger = spcm.Trigger(card)
    if USING_EXTERNAL_TRIGGER:
        trigger.or_mask(spcm.SPC_TMASK_EXT0)  # external trigger
        trigger.ext0_mode(spcm.SPC_TM_POS)
        trigger.ext0_level0(0.5 * units.V)
        trigger.ext0_coupling(spcm.COUPLING_DC)
        trigger.termination(1)  # 50 Ohm termination
    else:
        trigger.or_mask(spcm.SPC_TMASK_NONE)  # none trigger (using force trigger from software)

    # Setup the clock
    clock = spcm.Clock(card)
    clock.sample_rate(10 * units.percent)  # 10% of the maximum sample rate
    clock.clock_output(False)

    # generate the data and transfer it to the card
    setup_sequence(sequence)
    print("... setting up the sequence")

    # Test the step setup
    print(sequence)

    # We'll start and wait until all sequences are replayed.
    card.timeout(0) # no timeout
    print("Starting the card")
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_FORCETRIGGER)

    print("Sequence replay runs, switch to next sequence (3 times possible) with")
    if not USING_EXTERNAL_TRIGGER:
        print(" key: c ... change sequence")
    else:
        print(" a (slow) TTL signal on external trigger input connector")
    print(" key: ESC ... stop replay and end program")

    card_status = 0
    part_actual = 0
    sequence_status_old = 0
    sequence_status = 0

    part_transitions = [1, 3, 5, 7]

    while True:
        key = kb_hit()
        if key == 27:  # ESC
            card.stop()
            break

        elif key == ord('c') or key == ord('C'):
            if not USING_EXTERNAL_TRIGGER:
                print("sequence {0:d}".format(part_actual))
                part_last_step = sequence.steps[part_transitions[part_actual]]
                new_step =       sequence.steps[part_transitions[part_actual] + 1]
                part_last_step.update(to_step=new_step)
                part_actual += 1

        # end loop if card reports "ready" state, meaning that it has reached the end of the sequence
        if not card.is_demo_card():
            card_status = card.status()
            if (card_status & spcm.M2STAT_CARD_READY) != 0:
                break
        else:
            if part_actual >= len(part_transitions):
                break
    print("... Finished the sequence and stopping the card")


