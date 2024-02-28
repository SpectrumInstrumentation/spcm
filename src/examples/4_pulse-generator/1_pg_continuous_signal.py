"""
Spectrum Instrumentation GmbH (c)

1_pg_continuous_signal.py

This example sets up the pulse generator for X0 to create a continuous signal

Example for cards of the the M2p, M4i, M4x and M5i card-families (with pulse-generator add-on).

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm

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
    max_sample_rate = clock.max_sample_rate()
    sample_rate = clock.sample_rate(max_sample_rate)

    # enable pulse generator output on XIO lines
    multi_ios = spcm.MultiPurposeIOs(card)
    multi_ios[0].x_mode(spcm.SPCM_XMODE_PULSEGEN)

    # start the pulse generator
    # setup pulse generator 0 (output on X0)
    pulse_generators = spcm.PulseGenerators(card, spcm.SPCM_PULSEGEN_ENABLE0)
    # get the clock of the card
    pulse_gen_clock_Hz = pulse_generators.get_clock()

    # generate a continuous signal with 1 MHz
    len_1MHz = int(pulse_gen_clock_Hz / spcm.MEGA(1))
    pulse_generators[0].mode(spcm.SPCM_PULSEGEN_MODE_TRIGGERED)
    pulse_generators[0].period_length(len_1MHz)
    pulse_generators[0].high_length(len_1MHz // 2) # 50% HIGH, 50% LOW
    pulse_generators[0].delay(0)
    pulse_generators[0].num_loops(0) # 0: infinite
    pulse_generators[0].mux1(spcm.SPCM_PULSEGEN_MUX1_SRC_UNUSED)
    pulse_generators[0].mux2(spcm.SPCM_PULSEGEN_MUX2_SRC_SOFTWARE) # started by software force command

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

