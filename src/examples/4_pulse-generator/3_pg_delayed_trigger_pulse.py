"""
Spectrum Instrumentation GmbH (c)

3_pg_delayed_trigger_pulse.py

This functions creates a single pulse on X1 after the card has detected a trigger event.
Possible application: "change" length of trigger output signal, or create a delayed version of trigger output

Example for cards of the the M2p, M4i, M4x and M5i card-families (with pulse-generator add-on).

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
from spcm import units


card : spcm.Card

# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
# with spcm.Card(card_type=spcm.SPCM_TYPE_AO) as card:          # if you want to open the first card of a specific type
with spcm.Card('/dev/spcm0') as card:                           # if you want to open a specific card

    # first we set up the channel selection and the clock
    # for this example we enable only one channel to be able to use max sampling rate on all card types
    # ! changing the card settings while pulse generators are active causes a stop and restart of the pulse generators
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)

    # Setup the clock
    clock = spcm.Clock(card)
    clock.sample_rate(max = True)

    # Setup xio lines
    multi_ios = spcm.MultiPurposeIOs(card)
    multi_ios[0].x_mode(spcm.SPCM_XMODE_TRIGOUT)
    multi_ios[1].x_mode(spcm.SPCM_XMODE_PULSEGEN)

    # Setup trigger
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)

    # start the pulse generators
    pulse_generators = spcm.PulseGenerators(card, enable=spcm.SPCM_PULSEGEN_ENABLE1)
    pulse_gen_clock = pulse_generators.get_clock()

    # start the pulse generator
    # we use software trigger to trigger the card immediately after it has been started (start command at end of this function)
    len_1ms    = int(pulse_gen_clock * 0.001 + 1) # +1 because the HIGH area needs to be at least one sample less than length, so we increase length by one to get the calculated HIGH time
    offset_5us = int(pulse_gen_clock * 0.000005)
    pulse_generators[1].mode(spcm.SPCM_PULSEGEN_MODE_SINGLESHOT)
    pulse_generators[1].pulse_period(1 * units.ms) # or
    pulse_generators[1].pulse_length(0.99 * units.ms) # or
    pulse_generators[1].start_delay(0 * units.us)
    pulse_generators[1].repetitions(1) # just once
    pulse_generators[1].start_condition_state_signal(spcm.SPCM_PULSEGEN_MUX1_SRC_UNUSED)
    pulse_generators[1].start_condition_trigger_signal(spcm.SPCM_PULSEGEN_MUX2_SRC_CARDTRIGGER)

    # write the settings to the card
    # update the clock section to generate the programmed frequencies (SPC_SAMPLERATE)
    # and write the pulse generator settings
    pulse_generators.write_setup()

    # start card. Its trigger event will start the pulse generator
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER)

    # wait until user presses a key
    input("Press a key to stop the pulse generator(s) ")

    # stop the pulse generators
    pulse_generators.enable(False)
    pulse_generators.force()

