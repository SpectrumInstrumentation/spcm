"""
Spectrum Instrumentation GmbH (c)

3_gen_sequence.py

Shows a simple sequence mode example using only the few necessary commands

Example for analog replay cards (AWG) for the the M2p, M4i and M4x card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
from spcm import units

import numpy as np
from enum import IntEnum

import msvcrt
from time import sleep

USING_EXTERNAL_TRIGGER = False
LAST_STEP_OFFSET = 0

def kb_hit():
    """
    get the key that was pressed

    Returns
    -------
    int
        the ASCII code of the key that was pressed
    """
    return ord(msvcrt.getch()) if msvcrt.kbhit() else 0


def vWriteSegmentData (sequence : spcm.Sequence, segment_index : int, segment_len_sample : int):
    """
    transfers the data for a segment to the card's memory

    Parameters
    ----------
    sequence : spcm.Sequence
        the sequenc object that handles the data transfer
    segment_index : int
        the index of the segement to write
    segment_len_sample : int
        number of samples in the segment
    """

    # setup
    sequence.write_segment(segment_index)
    sequence.segment_size(segment_len_sample)

    # write data to board (main) sample memory
    sequence.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA, spcm.M2CMD_DATA_WAITDMA, transfer_length=segment_len_sample)


# (main) sample memory segment index:
class SEGMENT_IDX(IntEnum):
    SEG_RAMPUP   =  0  # ramp up
    SEG_RAMPDOWN =  1  # ramp down
    SEG_SYNC     =  2  # negative sync puls, for example oscilloscope trigger
    #                       3 // unused
    SEG_Q1SIN    =  4  # first quadrant of sine signal
    SEG_Q2SIN    =  5  # second quadrant of sine signal
    SEG_Q3SIN    =  6  # third quadrant of sine signal
    SEG_Q4SIN    =  7  # fourth quadrant of sine signal
    SEG_STOP     =  8  # DC level for stop/end


def vDoDataCalculation(sequence : spcm.Sequence):
    """
    calculates and writes the output data for all segments

    Parameters
    ----------
    sequence : spcm.Sequence
        the sequenc object that handles the data transfer
    """

    print("Calculation of output data")

    factor = 1
    # This series has a slightly increased minimum size value.
    series = sequence.card.series()
    if series in [spcm.TYP_M4IEXPSERIES, spcm.TYP_M4XEXPSERIES, spcm.TYP_M5IEXPSERIES]:
        factor = 6

    # buffer for data transfer
    segment_len_sample = factor * 512
    sequence.allocate_buffer(segment_len_sample)

    # helper values: Full Scale
    full_scale = sequence.card.max_sample_value()
    half_scale = full_scale // 2

    # !!! to keep the example simple we will generate the same data on all active channels !!!

    # data for the channels is interleaved. This means that we first write the first sample for each
    # of the active channels into the buffer, then the second sample for each channel, and so on
    # Please see the hardware manual, chapte "Data organization" for more information

    # --- sync puls: first half zero, second half -FS
    segment_len_sample = factor * 80
    sequence.buffer[:, :segment_len_sample // 2] = 0
    sequence.buffer[:, segment_len_sample // 2:] = -full_scale
    vWriteSegmentData(sequence, SEGMENT_IDX.SEG_SYNC, segment_len_sample)

    # --- ramp up
    segment_len_sample = factor * 64
    i = np.arange(segment_len_sample)
    sequence.buffer[:, i] = i * half_scale // segment_len_sample
    vWriteSegmentData(sequence, SEGMENT_IDX.SEG_RAMPUP, segment_len_sample)

    # --- ramp down
    segment_len_sample = factor * 64
    i = np.arange(segment_len_sample)
    sequence.buffer[:, i] = full_scale - (i * half_scale // segment_len_sample)
    vWriteSegmentData(sequence, SEGMENT_IDX.SEG_RAMPDOWN, segment_len_sample)

    # sine
    # write each quadrant in an own segment
    # --- sine, 1st quarter
    segment_len_sample = factor * 128
    i = np.arange(segment_len_sample)
    sequence.buffer[:, i] = half_scale + np.int32(half_scale * np.sin(2.0 * np.pi * (i + 0*segment_len_sample) / (segment_len_sample * 4)) + 0.5)
    vWriteSegmentData(sequence, SEGMENT_IDX.SEG_Q1SIN, segment_len_sample)

    # --- sine, 2nd quarter
    segment_len_sample = factor * 128
    i = np.arange(segment_len_sample)
    sequence.buffer[:, i] = half_scale + np.int32(half_scale * np.sin(2.0 * np.pi * (i + 1*segment_len_sample) / (segment_len_sample * 4)) + 0.5)
    vWriteSegmentData(sequence, SEGMENT_IDX.SEG_Q2SIN, segment_len_sample)

    # --- sine, 3rd quarter
    segment_len_sample = factor * 128
    i = np.arange(segment_len_sample)
    sequence.buffer[:, i] = half_scale + np.int32(half_scale * np.sin(2.0 * np.pi * (i + 2*segment_len_sample) / (segment_len_sample * 4)) + 0.5)
    vWriteSegmentData(sequence, SEGMENT_IDX.SEG_Q3SIN, segment_len_sample)

    # --- sine, 4th quarter
    segment_len_sample = factor * 128
    i = np.arange(segment_len_sample)
    sequence.buffer[:, i] = half_scale + np.int32(half_scale * np.sin(2.0 * np.pi * (i + 3*segment_len_sample) / (segment_len_sample * 4)) + 0.5)
    vWriteSegmentData(sequence, SEGMENT_IDX.SEG_Q4SIN, segment_len_sample)


    # --- DC level
    segment_len_sample = factor * 128
    sequence.buffer[:, :] = full_scale // 2
    vWriteSegmentData(sequence, SEGMENT_IDX.SEG_STOP, segment_len_sample)


def vConfigureSequence(sequence : spcm.Sequence):
    """
    vConfigureSequence

    Parameters
    ----------
    sequence : spcm.Sequence
        the sequenc object that handles the data transfer
    
    sequence memory
    four sequence loops are programmed (each with 6 steps)
    a keystroke or ext. trigger switched to the next sequence
    the loop value for the ramp increase in each sequence
     0 ...  5: sync, Q1sin, Q2sin, Q3sin, Q4sin, ramp up
     8 ... 13: sync, Q2sin, Q3sin, Q4sin, Q1sin, ramp down
    16 ... 21: sync, Q3sin, Q4sin, Q1sin, Q2sin, ramp up
    24 ... 29: sync, Q4sin, Q1sin, Q2sin, Q3sin, ramp down

                           +-- StepIndex
                           |   +-- StepNextIndex
                           |   |  +-- SegmentIndex
                           |   |  |                          +-- Loops
                           |   |  |                          |   +-- Flags: SPCSEQ_ENDLOOPONTRIG
       sine                |   |  |                          |   |          For using this flag disable Software-Trigger above."""
    sequence.step_memory(  0,  1, SEGMENT_IDX.SEG_SYNC,      3,  0)
    sequence.step_memory(  1,  2, SEGMENT_IDX.SEG_Q1SIN,     1,  0)
    sequence.step_memory(  2,  3, SEGMENT_IDX.SEG_Q2SIN,     1,  0)
    sequence.step_memory(  3,  4, SEGMENT_IDX.SEG_Q3SIN,     1,  0)
    sequence.step_memory(  4,  5, SEGMENT_IDX.SEG_Q4SIN,     1,  0)
    if USING_EXTERNAL_TRIGGER == False:
        sequence.step_memory(  5,  1,  SEGMENT_IDX.SEG_RAMPDOWN,  1,  0)
    else:
        sequence.step_memory(  5,  8,  SEGMENT_IDX.SEG_RAMPDOWN,  1,  spcm.SPCSEQ_ENDLOOPONTRIG)
    # all our sequences come in groups of five segments
    global LAST_STEP_OFFSET
    LAST_STEP_OFFSET = 5

    # cosine
    sequence.step_memory(  8,  9, SEGMENT_IDX.SEG_SYNC,      3,  0)
    sequence.step_memory(  9, 10, SEGMENT_IDX.SEG_Q2SIN,     1,  0)
    sequence.step_memory( 10, 11, SEGMENT_IDX.SEG_Q3SIN,     1,  0)
    sequence.step_memory( 11, 12, SEGMENT_IDX.SEG_Q4SIN,     1,  0)
    sequence.step_memory( 12, 13, SEGMENT_IDX.SEG_Q1SIN,     1,  0)
    if USING_EXTERNAL_TRIGGER == False:
        sequence.step_memory( 13,  9,  SEGMENT_IDX.SEG_RAMPUP,    2,  0)
    else:
        sequence.step_memory( 13, 16,  SEGMENT_IDX.SEG_RAMPUP,    2,  spcm.SPCSEQ_ENDLOOPONTRIG)

    # inverted sine
    sequence.step_memory( 16, 17, SEGMENT_IDX.SEG_SYNC,      3,  0)
    sequence.step_memory( 17, 18, SEGMENT_IDX.SEG_Q3SIN,     1,  0)
    sequence.step_memory( 18, 19, SEGMENT_IDX.SEG_Q4SIN,     1,  0)
    sequence.step_memory( 19, 20, SEGMENT_IDX.SEG_Q1SIN,     1,  0)
    sequence.step_memory( 20, 21, SEGMENT_IDX.SEG_Q2SIN,     1,  0)
    if USING_EXTERNAL_TRIGGER == False:
        sequence.step_memory( 21, 17,  SEGMENT_IDX.SEG_RAMPDOWN,  3,  0)
    else:
        sequence.step_memory( 21, 24,  SEGMENT_IDX.SEG_RAMPDOWN,  3,  spcm.SPCSEQ_ENDLOOPONTRIG)

    # inverted cosine
    sequence.step_memory( 24, 25, SEGMENT_IDX.SEG_SYNC,      3,  0)
    sequence.step_memory( 25, 26, SEGMENT_IDX.SEG_Q4SIN,     1,  0)
    sequence.step_memory( 26, 27, SEGMENT_IDX.SEG_Q1SIN,     1,  0)
    sequence.step_memory( 27, 28, SEGMENT_IDX.SEG_Q2SIN,     1,  0)
    sequence.step_memory( 28, 29, SEGMENT_IDX.SEG_Q3SIN,     1,  0)
    sequence.step_memory( 29, 30, SEGMENT_IDX.SEG_RAMPUP,    4,  0)
    sequence.step_memory( 30, 30, SEGMENT_IDX.SEG_STOP,      1,  spcm.SPCSEQ_END)  # M2i: only a few sample from this segment are replayed
                                                                       # M4i: the complete segment is replayed

    # Configure the beginning (index of first seq-entry to start) of the sequence replay.
    sequence.start_step(0)

    if True:
        for i in range(0, 32, 1):
            next_step_index, seqment_index, loops, flags = sequence.step_memory(i)
            print("Step {}: next {}, segment {}, loops {}, flags 0b{:b}".format(i, next_step_index, seqment_index, loops, abs(flags)))


card : spcm.Card
# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(card_type=spcm.SPCM_TYPE_AO) as card:             # if you want to open the first card of a specific type
    
    # setup card mode
    card.card_mode(spcm.SPC_REP_STD_SEQUENCE)
    
    # set up the channels
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    channels.enable(True)
    channels.amp(1 * units.V)
    channels.stop_level(spcm.SPCM_STOPLVL_HOLDLAST)

    # set up the mode
    max_segments = 32
    sequence = spcm.Sequence(card)
    sequence.max_segments(max_segments)

    # set up trigger
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)  # software trigger

    # Setup the clock
    clock = spcm.Clock(card)
    series = card.series()
    if (series in [spcm.TYP_M4IEXPSERIES, spcm.TYP_M4XEXPSERIES]):
        sample_rate = clock.sample_rate(50 * units.MHz, return_unit=units.MHz)
    else:
        sample_rate = clock.sample_rate(1 * units.MHz, return_unit=units.MHz)
    clock.clock_output(0)

    # generate the data and transfer it to the card
    vDoDataCalculation(sequence)
    print("... data has been transferred to board memory")

    # define the sequence in which the segments will be replayed
    vConfigureSequence(sequence)
    print("... sequence configured")

    # We'll start and wait until all sequences are replayed.
    card.timeout(0 * units.s)
    print("Starting the card")
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER)

    print("sequence replay runs, switch to next sequence (3 times possible) with")
    if USING_EXTERNAL_TRIGGER is False:
        print(" key: c ... change sequence")
    else:
        print(" a (slow) TTL signal on external trigger input connector")
    print(" key: ESC ... stop replay and end program")

    card_status = 0
    sequence_actual = 0    # first step in a sequence
    sequence_next = 0
    sequence_status_old = 0

    while True:
        key = kb_hit()
        if key == 27:  # ESC
            card.stop()
            break

        elif key == ord('c') or key == ord('C'):
            if USING_EXTERNAL_TRIGGER is False:
                sequence_next = ((sequence_actual + 8) % 32)
                print("sequence {0:d}".format(sequence_next // 8))

                # switch to next sequence
                # (before it is possible to overwrite the segment data of the new used segments with new values)
                step = 0

                # --- change the next step value from the sequence end entry in the actual sequence
                next_step_index, segment_index, loops, flags = sequence.step_memory(sequence_actual + LAST_STEP_OFFSET)
                sequence.step_memory(sequence_actual + LAST_STEP_OFFSET, sequence_next, segment_index, loops, flags)

                sequence_actual = sequence_next
        else:
            sleep(0.01)  # 10 ms

            # Demonstrate the two different sequence status values at M2i and M4i / M2p cards.
            sequence_status = sequence.status()

            # print the status only when using external trigger to switch sequences
            if USING_EXTERNAL_TRIGGER:
                if sequence_status_old != sequence_status:
                    sequence_status_old = sequence_status

                    # Valid values only at a startet card available.
                    if card_status & spcm.M2STAT_CARD_PRETRIGGER:
                        print("status: actual sequence number: {0:d}".format(sequence_status))

        # end loop if card reports "ready" state, meaning that it has reached the end of the sequence
        card_status = card.status()
        if (card_status & spcm.M2STAT_CARD_READY) != 0:
            break


