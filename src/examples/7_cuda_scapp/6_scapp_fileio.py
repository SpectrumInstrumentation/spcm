"""
Spectrum Instrumentation GmbH (c)

6_scapp_fileio.py

Example that shows how to move data from the card through RDMA to the GPU and then directly write to disk.

PLEASE NOTE: you'll have to install mamba to use kvikio and install the spcm examples requirements inside the 
kvikio-env generated

For analog recording cards (digitizers) for the the M2p, M4i, M4x and M5i card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""


import spcm
from spcm import units

import os
import kvikio

card : spcm.Card


# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(card_type=spcm.SPCM_TYPE_AI) as card:            # if you want to open the first card of a specific type

    # do a simple standard setup
    card.card_mode(spcm.SPC_REC_FIFO_SINGLE)     # single trigger standard mode
    card.timeout(5 * units.s)

    # setup trigger engine
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)

    # setup channels
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    channels[0].amp(1 * units.V)

    # we try to use the max samplerate
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)
    sample_rate = clock.sample_rate(6.4 * units.GHz, return_unit=(units.MHz))
    print(f"Used Sample Rate: {sample_rate}")
    
    # Setup a data transfer object with CUDA DMA
    num_samples    =    1 * units.GiS # MebiSamples = 1024*1024 Samples
    notify_samples =  128 * units.MiS # KibiSamples = 1024 Samples
    notify_samples_magnitude = notify_samples.to_base_units().magnitude

    scapp_transfer = spcm.SCAPPTransfer(card, direction=spcm.Direction.Acquisition)
    scapp_transfer.notify_samples(notify_samples)
    scapp_transfer.allocate_buffer(num_samples)
    scapp_transfer.auto_avail_card_len(False) # handle triggering new data manually
    scapp_transfer.start_buffer_transfer()
    
    num_threads = 8

    # Path to a file 
    # Make sure that the file is located on a disk that is fast enough to keep up with the data rate
    path = "scapp_fileio_test.bin"

    with kvikio.CuFile(path, "w") as f:
        try:
            print("Starting card...")
            counter = 0
            offset = 0
            output_divider = 200 # print memory fill every 200 data blocks
            futures = []

            card.start(spcm.M2CMD_CARD_ENABLETRIGGER | spcm.M2CMD_DATA_STARTDMA)
            for data_raw_gpu in scapp_transfer:
                if counter < num_threads:
                    futures.append(f.pwrite(data_raw_gpu, file_offset=offset))
                else:
                    while not futures[counter % num_threads].done():
                        pass
                    scapp_transfer.avail_card_len(notify_samples_magnitude)
                    futures[counter % num_threads] = f.pwrite(data_raw_gpu, file_offset=offset)
                offset += notify_samples_magnitude
                if counter % output_divider == 0:
                    print("fill size: {:>4d} promille".format(scapp_transfer.fill_size_promille()), end="\r")
                counter += 1
        except KeyboardInterrupt:
            print("Ctrl+C pressed. Stopped recording.")
        except Exception as e:
            print(e)
        [future.get() for future in futures]

    print("file size: {} GiB".format(os.path.getsize(path)/1024/1024/1024))

        




