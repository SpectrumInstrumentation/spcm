"""
Spectrum Instrumentation GmbH (c)

2_pg_multiple_continuous_signals.py

Different setups are shown to highlight the capabilities of the pulse generator feature.

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

    clock = spcm.Clock(card)
    sample_rate = clock.sample_rate(max = True, return_unit=units.MHz)

    multi_ios = spcm.MultiPurposeIOs(card)
    multi_ios[0].x_mode(spcm.SPCM_XMODE_PULSEGEN)
    multi_ios[1].x_mode(spcm.SPCM_XMODE_PULSEGEN)
    multi_ios[2].x_mode(spcm.SPCM_XMODE_PULSEGEN)
    multi_ios[3].x_mode(spcm.SPCM_XMODE_PULSEGEN)

    # start the pulse generator 0
    pulse_generators = spcm.PulseGenerators(card, enable=True)
    pulse_gen_clock = pulse_generators.get_clock()

    # setup pulse generator 0 (output on X0)
    # generate a continuous signal with the maximum frequency (in units of clock ticks)
    pulse_generators[0].mode(spcm.SPCM_PULSEGEN_MODE_TRIGGERED)
    pulse_generators[0].period_length(2)
    pulse_generators[0].high_length(1) # 50% HIGH, 50% LOW
    pulse_generators[0].delay(0)
    pulse_generators[0].num_loops(0) # 0: infinite
    pulse_generators[0].mux1(spcm.SPCM_PULSEGEN_MUX1_SRC_UNUSED)
    pulse_generators[0].mux2(spcm.SPCM_PULSEGEN_MUX2_SRC_SOFTWARE) # started by software force command


    # setup the pulse generator 1 (output on x1)
    # generate a continuous signal with ~1 MHz, 50% duty cycle
    pulse_generators[1].mode(spcm.SPCM_PULSEGEN_MODE_TRIGGERED)
    pulse_generators[1].pulse_period(1 * units.us) # or
    # pulse_generators[1].repetition_rate(1 * units.MHz)
    pulse_generators[1].pulse_length(0.5 * units.us) # or
    # pulse_generators[1].duty_cycle(50 * units.percent)
    pulse_generators[1].start_delay(0 * units.us)
    pulse_generators[1].repetitions(0) # 0: infinite
    pulse_generators[1].start_condition_state_signal(spcm.SPCM_PULSEGEN_MUX1_SRC_UNUSED)
    pulse_generators[1].start_condition_trigger_signal(spcm.SPCM_PULSEGEN_MUX2_SRC_SOFTWARE)


    # setup the pulse generator 2 (output on x2)
    # same signal as pulse generator 1, but with a phase shift
    pulse_generators[2].mode(spcm.SPCM_PULSEGEN_MODE_TRIGGERED)
    pulse_generators[2].pulse_period(1 * units.us) # or
    # pulse_generators[2].repetition_rate(1 * units.MHz)
    pulse_generators[2].pulse_length(0.5 * units.us) # or
    # pulse_generators[2].duty_cycle(50 * units.percent)
    pulse_generators[2].start_delay(0.25 * units.us)
    pulse_generators[2].repetitions(0) # 0: infinite
    pulse_generators[2].start_condition_state_signal(spcm.SPCM_PULSEGEN_MUX1_SRC_UNUSED)
    pulse_generators[2].start_condition_trigger_signal(spcm.SPCM_PULSEGEN_MUX2_SRC_SOFTWARE)


    if len(multi_ios) > 3:
        # setup the pulse generator 3 (output on x3. Not available on M4i/M4x)
        # generate a continuous signal with ~500 kHz after the first edge on pulse generator 2 occurred, and delay the start for two periods of the 1MHz signal.
        len_500kHz = int(pulse_gen_clock / spcm.KILO(500))
        pulse_generators[3].mode(spcm.SPCM_PULSEGEN_MODE_TRIGGERED)
        pulse_generators[3].pulse_period(2 * units.us) # or
        # pulse_generators[3].repetition_rate(500 * units.kHz)
        pulse_generators[3].pulse_length(1.8 * units.us) # or
        # pulse_generators[3].duty_cycle(90 * units.percent)
        pulse_generators[3].start_delay(2 * units.us)
        pulse_generators[3].repetitions(0) # 0: infinite
        pulse_generators[3].start_condition_state_signal(spcm.SPCM_PULSEGEN_MUX1_SRC_UNUSED)
        pulse_generators[3].start_condition_trigger_signal(spcm.SPCM_PULSEGEN_MUX2_SRC_PULSEGEN2) # started by first edge of pulse generator 2

    # write the settings to the card
    # update the clock section to generate the programmed frequencies (SPC_SAMPLERATE)
    # and write the pulse generator settings
    pulse_generators.write_setup()

    # start all pulse generators that wait for a software command
    pulse_generators.force()

    # wait until user presses a key
    input("Press a key to stop the pulse generator(s) ")

    # stop the pulse generators
    pulse_generators.enable(False)
    pulse_generators.write_setup()

