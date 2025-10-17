# General imports
import sys
import time

# Imports for the calculation
import cupy as cp

# Plotting
import matplotlib
matplotlib.use('qtagg') # Use Tkinter-based backend
import matplotlib.pyplot as plt

# # Save the CUDA source code for debugging you can find the file in the directory: ~/.cupy/kernel_cache
# import os
# os.environ["CUPY_CACHE_SAVE_CUDA_SOURCE"] = "1"

# Imports for the SPCM card and the phase noise calculation
import spcm
from spcm import units

# Settings
num_averages = 50
num_iterations = 1000


with spcm.Card(card_type=spcm.SPCM_TYPE_AI, verbose=False) as card:
    card_info = card.product_name()
    print(f"Card: {card_info}")

    # do a fifo single setup
    card.card_mode(spcm.SPC_REC_FIFO_MULTI)
    card.timeout(50 * units.s)

    # setup trigger engine
    trigger = spcm.Trigger(card)
    # trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)
    trigger.or_mask(spcm.SPC_TMASK_EXT0)
    trigger.ext0_mode(spcm.SPC_TM_POS)
    trigger.ext0_level0(0.5 * units.V)
    trigger.ext0_coupling(spcm.COUPLING_DC)
    trigger.termination(0)

    # setup channels
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)  # enable channel 0
    num_channels = len(channels)
    amplitude = channels.amp(0.5 * units.V, return_unit=units.V)
    max_value = card.max_sample_value()

    # we try to use the max samplerate
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)
    sample_rate = clock.sample_rate(10 * units.GHz, return_unit=units.MHz)
    sample_rate_magnitude = sample_rate.to_base_units().magnitude 
    print(f"Used Sample Rate: {sample_rate}")
    
    # Setup a data transfer object with CUDA DMA
    notify_samples = 10 * units.MiS
    notify_samples_magnitude = int(notify_samples.to_base_units().magnitude)
    reduced_samples_magnitude = int(notify_samples_magnitude / 1.5)

    num_segments = 32
    buffer_factor = 4

    scapp_transfer = spcm.SCAPPMulti(card)
    scapp_transfer.data_conversion(spcm.SPCM_DC_12BIT_TO_12BITPACKED)
    scapp_transfer.notify_samples(notify_samples)
    scapp_transfer.allocate_buffer(segment_samples=notify_samples, num_segments=num_segments*buffer_factor)
    scapp_transfer.post_trigger(notify_samples - 64)
    scapp_transfer.start_buffer_transfer()


    # Allocate memory on GPU
    data_unpacked_gpu = cp.zeros((num_channels, notify_samples_magnitude), order='F', dtype=cp.int16)

    num_threads = 1024
    num_blocks = (notify_samples_magnitude * num_channels) // num_threads

    
    print("Compiling the kernel...")
    t0 = time.time_ns()
    kernel_unpack_12bit = cp.RawKernel(r'''
        extern "C" __global__
        void kernel_unpack_12bit(const char* input, short* output) {
            // each thread loads one element from global to shared mem
            unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned long j = 0;
            short nNibble_H, nNibble_M, nNibble_L;
            
            if (i % 2 == 0) {
                j = i / 2 * 3;
                nNibble_H = (input[j + 1] >> 0) & 0xF;
                nNibble_M = (input[j + 0] >> 4) & 0xF;
                nNibble_L = (input[j + 0] >> 0) & 0xF;
                output[i] = (static_cast < short > (nNibble_H << 12) >> 4) | (nNibble_M << 4) | (nNibble_L << 0);
            } else {
                j = (i - 1) / 2 * 3 + 1;
                nNibble_H = (input[j + 1] >> 4) & 0xF;
                nNibble_M = (input[j + 1] >> 0) & 0xF;
                nNibble_L = (input[j + 0] >> 4) & 0xF;
                output[i] = (static_cast < short > (nNibble_H << 12) >> 4) | (nNibble_M << 4) | (nNibble_L << 0);
            }
            
        }''', 'kernel_unpack_12bit')
    kernel_unpack_12bit.compile(log_stream=sys.stdout)
    print("...finished compiling the kernel (took: {:.3} ms)".format((time.time_ns() - t0) / 1e6))

    counter = 0
    average_counter = 0
    
    print("GPU memory - initial usage: {:.3} GB free".format(cp.cuda.runtime.memGetInfo()[0]/1024/1024/1024))

    card.start(spcm.M2CMD_CARD_ENABLETRIGGER | spcm.M2CMD_DATA_STARTDMA)
    try:
        for data_raw_gpu in scapp_transfer:
            # waits for a block to become available after the data is transferred from the card to the gpu memory using scapp

            kernel_unpack_12bit((num_blocks,), (num_threads,), (data_raw_gpu, data_unpacked_gpu))
            
            data_block_sum_gpu += data_unpacked_gpu

            counter += 1
            if counter % num_averages == 0:
                average_counter += 1

                print(f"\nAverage counter: {average_counter}\n---")
                print(f"Minimum value avg: {data_block_sum_gpu.min()/num_averages}")
                print(f"Maximum value avg: {data_block_sum_gpu.max()/num_averages}")

                # Monitor the GPU memory usage
                # print("GPU memory - after iteration {:2}: {:.3} GB free".format(average_counter, cp.cuda.runtime.memGetInfo()[0]/1024/1024/1024))

                data_block_sum_cpu = cp.asnumpy(data_block_sum_gpu)
                data_block_sum_gpu[:] = 0

            if average_counter >= num_iterations:
                break

    except spcm.SpcmException as e:
        print(f"{e}")
    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    plt.figure()
    plt.plot(data_block_sum_cpu / num_averages)
    plt.title(f"Averaged signal (factor: {num_averages})")
    plt.xlabel("Sample number")
    plt.ylabel("Amplitude [LSB]")
    plt.grid()
    plt.show()

    print("Main loop finished")

