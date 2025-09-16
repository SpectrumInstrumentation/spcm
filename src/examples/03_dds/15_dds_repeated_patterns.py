"""  
Spectrum Instrumentation GmbH (c) 2024

15_dds_repeated_patterns.py

Repeats a pattern of DDS commands multiple times. It is possible to change the frequency and amplitude of the DDS signal in each iteration of the pattern.

Example for analog replay cards (AWG) for the the M4i and M4x card-families with installed DDS option.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
from spcm import units

import time

card : spcm.Card
# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(card_type=spcm.SPCM_TYPE_AO) as card:             # if you want to open the first card of a specific type

    START_PATTERN_WITH_EXT_TRIGGER = False  # Set to True if you want to start the pattern with an external trigger, otherwise it will be started with a force trigger

    # setup card for DDS
    card.card_mode(spcm.SPC_REP_STD_DDS)

    # Setup the channels
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    channels[0].enable(True)
    channels[0].output_load(50 * units.ohm)
    channels[0].amp(500 * units.mV)

    # Setup the trigger engine
    trigger = spcm.Trigger(card, channels=channels)
    if START_PATTERN_WITH_EXT_TRIGGER:
        trigger.or_mask(spcm.SPC_TMASK_EXT0) # Enable external trigger source, so the pattern can be started with an external trigger
        trigger.ext0_coupling(spcm.COUPLING_DC)  # Set the external trigger coupling to DC
        trigger.ext0_mode(spcm.SPC_TM_POS)   # Set the external trigger mode
        trigger.ext0_level0(1.5 * units.V)  # Set the external trigger level to 1.5 V
        trigger.termination(1)  # Set the termination to 1 (50 Ohm)
    else:
        trigger.or_mask(spcm.SPC_TM_NONE) # Disable all trigger sources and use force triggers to control the DDS execution
    
    card.write_setup() # IMPORTANT! this turns on the card's system clock signals, that are required for DDS to work

    # Setup DDS functionality
    dds = spcm.DDS(card, channels=channels)
    dds.reset()

    # Initialize the DDS cores

    dds[0].amp(0)   # Set amplitude to 0 dBm
    dds[0].freq(100 * units.Hz)  # Set frequency to 100 Hz
    dds[0].phase(0)
    
    dds.exec_at_trg()
    dds.write_to_card()

    # Start the card and the DDS firmware using the card's trigger engine
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_FORCETRIGGER)

    # Create a loop to run through 10 different patterns
    for pattern_nr in range(10):
        print(f"Sending pattern {pattern_nr} to the FIFO...")

        # Step 0 in the pattern
        dds.trg_src(spcm.SPCM_DDS_TRG_SRC_TIMER)

        dds.trg_timer(0.01)
        dds[0].freq(100 * units.Hz + pattern_nr * 10 * units.Hz)  # Set frequency to 100 kHz + pattern_nr * 100 kHz
        dds[0].amp(0.1)  # Set amplitude
        dds.exec_at_trg() # This is executed using either an external trigger or a force trigger, depending on the START_PATTERN_WITH_EXT_TRIGGER setting

        # Step 1 in the pattern
        dds.trg_timer(0.02)
        dds[0].freq(200 * units.Hz + pattern_nr * 10 * units.Hz)  # Set frequency to 100 kHz + pattern_nr * 100 kHz
        dds[0].amp(0.2)  # Set amplitude
        dds.exec_at_trg() # these are executed when the timer triggers, which is set to 10 ms in the first step

        # Step 2 in the pattern
        dds.trg_timer(0.03)
        dds[0].freq(300 * units.Hz + pattern_nr * 10 * units.Hz)  # Set frequency to 100 kHz + pattern_nr * 100 kHz
        dds[0].amp(0.3)  # Set amplitude
        dds.exec_at_trg() # these are executed when the timer triggers, which is set to 20 ms in the previous step

        # Step 3 in the pattern
        dds.trg_timer(0.04)
        dds[0].freq(400 * units.Hz + pattern_nr * 10 * units.Hz)  # Set frequency to 100 kHz + pattern_nr * 100 kHz
        dds[0].amp(0.4)  # Set amplitude
        dds.exec_at_trg() # these are executed when the timer triggers, which is set to 30 ms in the previous step

        # Final step in the pattern
        dds[0].amp(0.0)  # Turn everything off
        
        dds.trg_src(spcm.SPCM_DDS_TRG_SRC_CARD) # Disable the timer and wait for the card trigger
        dds.exec_at_trg() # these are executed when the timer triggers, which is set to 40 ms in the previous step

        dds.write_to_card()

        # The full pattern is transferred to the FIFO of the DDS firmware, now we start the execution of patterns using a force trigger
        # NOTE: This trigger triggers the first exec_at_trg() command in the command list above.
        if START_PATTERN_WITH_EXT_TRIGGER:
            print("Waiting for external trigger to start the pattern...")
        else:
            print(f"Starting pattern {pattern_nr} with force trigger...")
            trigger.force()
        print(f"Pattern {pattern_nr} started, waiting for FIFO to be emptied...")

        # While this while loop is running, the DDS firmware is (waiting for an external trigger to go) executing the commands in the FIFO.
        # You can have any waiting criteria here, for example you can wait until a measurement result is available, or until a specific condition is met.
        # In this example we just wait until the command counter is 0, which means that all commands in the FIFO have been executed.
        while True:
            command_counter = dds.queue_cmd_count()
            print(f"Command count: {command_counter:04d}", end='\r')  # print the command count in the same line
            if command_counter < 1:
                break
            time.sleep(0.1)  # wait for 100 milliseconds to let the DDS run
        print(f"\nPattern {pattern_nr} finished...")
        print("----------------------")

        time.sleep(1)  # wait for 1 second before sending the next pattern

    card.stop()

    # input("Press Enter to Exit")
