"""
Spectrum Instrumentation GmbH (c)

5_scapp_gen_fifo_invert.py

Example that shows how to combine the CUDA DMA transfer with the generation of data. The example uses FIFO replay mode
to generate data then send the data from the CPU to the GPU, which inverts the data and sends it to the card using RDMA
from the SCAPP option.

For analog recording cards (digitizers) for the the M2p, M4i, M4x and M5i card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""


import spcm
from spcm import units

import numpy as np
import matplotlib.pyplot as plt

card : spcm.Card

# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(card_type=spcm.SPCM_TYPE_AO, verbose=False) as card:            # if you want to open the first card of a specific type

    # do a simple standard setup
    card.card_mode(spcm.SPC_REP_FIFO_SINGLE)     # single trigger standard mode
    card.timeout(5 * units.s)

    # setup trigger engine
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)

    # setup channels
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    channels.enable(True)
    channels.output_load(50 * units.ohm)
    channels.amp(1 * units.V)

    # we try to use the max samplerate
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)
    sample_rate = clock.sample_rate(1 * units.MHz, return_unit=(units.MHz))
    print(f"Used Sample Rate: {sample_rate}")
    
    # Setup a data transfer object with CUDA DMA
    notify_samples = spcm.KIBI(128)
    num_samples    = spcm.MEBI(1)

    cuda_object   = spcm.CUDA(card, channels, 0, scapp=True)
    cuda_transfer = spcm.CUDATransfer(card, cuda_object, direction=spcm.CUDATransfer.Direction.Generation)

    cuda_transfer.notify_samples(notify_samples)

    # setup the kernel
    threads_per_block = 1024
    num_blocks = notify_samples // threads_per_block

    # CUDA kernel
    CudaKernelInvert = cuda_object.create_kernel(function_name="CudaKernelInvert", src = '''\
        extern "C" __global__ void CudaKernelInvert (short* pcIn, short* pcOut) 
            {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            pcOut[i] = -1 * pcIn[i]; 
            }
        ''', num_blocks=num_blocks, threads_per_block=threads_per_block)

    cuda_transfer.allocate_buffer(num_samples)
    cuda_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)
    # cuda_transfer.start_buffer_transfer()

    # card.start(spcm.M2CMD_CARD_ENABLETRIGGER | spcm.M2CMD_DATA_STARTDMA)
    
    time_range = np.arange(notify_samples) / sample_rate
    full_time_range = np.arange(num_samples) / sample_rate
    # plot function
    # fig, ax = plt.subplots()
    # # line1, = ax.plot(time_range, np.zeros_like(time_range))
    # # line2, = ax.plot(time_range, np.zeros_like(time_range))
    # line1, = ax.plot(full_time_range, np.zeros_like(full_time_range))
    # ax.set_ylim([-100, 100])  # range of Y axis
    # ax.xaxis.set_units(units.us)
    # ax.yaxis.set_units(units.mV)
    # plt.show(block=False)
    # plt.draw()

    # Pre-load data
    repeats = num_samples // notify_samples
    frequency = 1 * units.kHz
    # cuda_transfer.buffer[:] = np.sin(2*np.pi*full_time_range*2e4*units.Hz) * 0.1 * 32767
    # cuda_transfer.avail_card_len(notify_samples)
    cuda_transfer.cpu_to_gpu_buffer.array[:] = np.asarray(np.sin(2*np.pi*time_range*frequency) * 0.5 * 32767, dtype=np.int16)
    for i in range(repeats):
        cuda_transfer.cpu_to_gpu_buffer.copy_to_device()

        cuda_transfer.card_gpu_buffer.set_view(i*notify_samples, notify_samples)

        # start kernel on the GPU to process the transfered data
        kernel_arguments = (cuda_transfer.cpu_to_gpu_buffer, cuda_transfer.card_gpu_buffer)
        CudaKernelInvert.launch(kernel_arguments)

        cuda_transfer.avail_card_len(notify_samples)
        # cuda_transfer.card_gpu_buffer.set_view(i*notify_samples, notify_samples)
        # start kernel on the GPU to process the transfered data
        # kernel_arguments = (cuda_transfer.cpu_to_gpu_buffer, cuda_transfer.card_gpu_buffer)
        # CudaKernelInvert.launch(kernel_arguments)
        # cuda_transfer.card_gpu_buffer.copy_to_host()
        # print(cuda_transfer.card_gpu_buffer.view[:10], np.max(cuda_transfer.card_gpu_buffer.view), np.min(cuda_transfer.card_gpu_buffer.view))
    # line1.set_ydata(cuda_transfer.card_gpu_buffer.view)
    # line2.set_ydata(cuda_transfer.cpu_to_gpu_buffer.view)
    # cuda_transfer.card_gpu_buffer.copy_to_host()
    # line1.set_ydata(cuda_transfer.card_gpu_buffer.array)
    # fig.canvas.draw()
    # fig.canvas.flush_events()

    cuda_transfer.wait_dma()
    print("Pre-loaded data")

    plt.show(block=True)

    # card.cmd(spcm.M2CMD_DATA_STARTDMA)
    # card.start(spcm.M2CMD_CARD_ENABLETRIGGER | spcm.M2CMD_DATA_STARTDMA)
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER)

    try:
        # for card_gpu, cpu_gpu, gpu_cpu in cuda_transfer:
        for gpu_to_card, cpu_to_gpu in cuda_transfer:
            # this is the point to do anything with the data on the GPU

            # gpu_cpu.array[:] = np.sin(2*np.pi*time_range*2e5*units.Hz) * 100
            cpu_to_gpu.copy_to_device()

            # start kernel on the GPU to process the transfered data
            kernel_arguments = (cpu_to_gpu, gpu_to_card)
            CudaKernelInvert.launch(kernel_arguments)
            
            # after kernel has finished we copy processed data from GPU to host
            # gpu_to_card.copy_to_host()
            # print(gpu_to_card.view[:10], np.max(gpu_to_card.view), np.min(gpu_to_card.view))
            # print(cpu_to_gpu.view[:10], np.max(cpu_to_gpu.view), np.min(cpu_to_gpu.view))
            cuda_transfer.avail_card_len(notify_samples)

            print("Fillsize: {}".format(cuda_transfer.fill_size_promille()))
    
            # now the processed data is in the host memory
            # line1.set_ydata(card_gpu.view)
            # line1.set_ydata(card_gpu.array)
            # line2.set_ydata(gpu_cpu.view)
            # fig.canvas.draw()
            # fig.canvas.flush_events()
            # time.sleep(0.001)
    except KeyboardInterrupt:
        print("Ctrl+C pressed and generation stopped")






