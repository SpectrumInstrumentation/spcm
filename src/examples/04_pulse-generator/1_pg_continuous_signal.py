"""
Spectrum Instrumentation GmbH (c)

1_pg_continuous_signal.py

This example sets up the pulse generator for X0 to create a continuous signal

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
    sample_rate = clock.sample_rate(max = True)

    # enable pulse generator output on XIO line 0
    multi_io = spcm.MultiPurposeIO(card, 0)
    multi_io.x_mode(spcm.SPCM_XMODE_PULSEGEN)

    # start the pulse generator
    # setup pulse generator 0 (output on X0)
    pulse_generators = spcm.PulseGenerators(card, spcm.SPCM_PULSEGEN_ENABLE0)
    pulse_generator = pulse_generators[0]

    pulse_generator.mode(spcm.SPCM_PULSEGEN_MODE_TRIGGERED)
    pulse_period = pulse_generator.pulse_period(1 * units.us)
    rep_rate     = pulse_generator.repetition_rate(500 * units.kHz)
    pulse_length = pulse_generator.pulse_length(1.5 * units.us)
    duty_cycle   = pulse_generator.duty_cycle(50 * units.percent)
    start_delay  = pulse_generator.start_delay(3 * units.ms)
    repetitions  = pulse_generator.repetitions(0) # 0: infinite
    state_signal = pulse_generator.start_condition_state_signal(spcm.SPCM_PULSEGEN_MUX1_SRC_UNUSED)
    trg_signal   = pulse_generator.start_condition_trigger_signal(spcm.SPCM_PULSEGEN_MUX2_SRC_SOFTWARE)
    invert       = pulse_generator.invert_start_condition(False)

    print(f"Pulse period:            {pulse_period}")
    print(f"Repetition rate:         {rep_rate}")
    print(f"Pulse length:            {pulse_length}")
    print(f"Duty cycle:              {duty_cycle}")
    print(f"Start delay:             {start_delay}")
    print(f"Number of repetitions:   {repetitions}")
    print(f"Start condition state:   {state_signal}")
    print(f"Start condition trigger: {trg_signal}")
    print(f"Invert start condition:  {invert}")


    # write the settings to the card
    # update the clock section to generate the programmed frequencies (SPC_SAMPLERATE)
    # and write the pulse generator settings
    pulse_generators.write_setup()

    # start all pulse generators that wait for a software command
    pulse_generators.force()

    card.start()

    # wait until user presses a key
    input("Press a key to stop the pulse generator(s) ")

    # stop the pulse generators
    pulse_generators.enable(False)
    pulse_generators.write_setup()

