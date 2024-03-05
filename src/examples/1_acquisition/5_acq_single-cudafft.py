"""
Spectrum Instrumentation GmbH (c)

5_acq_single-cudafft.py

Shows a simple Standard mode example using only the few necessary commands, with the additional of a CUDA processing step to do an FFT on the GPU

Example for analog recording cards (digitizers) for the the M2p, M4i, M4x and M5i card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
import numpy as np
import cupy
import matplotlib.pyplot as plt


card : spcm.Card

# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(card_type=spcm.SPCM_TYPE_AI) as card:            # if you want to open the first card of a specific type

    # do a simple standard setup
    card.card_mode(spcm.SPC_REC_STD_SINGLE)     # single trigger standard mode
    card.timeout(5000)                   # timeout 5 s

    # setup trigger engine
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)  # trigger set to software
    trigger.and_mask(spcm.SPC_TMASK_NONE)

    # setup channels
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    amplitude_mV = 1000
    channels.amp(amplitude_mV)  # set input range to +/- 1000 mV

    # we try to use the max samplerate
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)            # clock mode internal PLL
    clock.output(0)                           # no clock output
    sample_rate = clock.sample_rate(max = True) # set to maximum sample rate
    print("Used samplerate: {0} MS/s\n".format(sample_rate // 1000000))

    # setup a data transfer buffer
    num_samples = spcm.KIBI(14)
    data_transfer = spcm.DataTransfer(card)
    data_transfer.memory_size(num_samples)
    data_transfer.allocate_buffer(num_samples)
    data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)

    # start card and DMA
    try:
        card.start(spcm.M2CMD_CARD_ENABLETRIGGER)
    except spcm.SpcmTimeout as timeout:
        print("... Timeout")

    try:
        data_transfer.wait()
    except spcm.SpcmTimeout as timeout:
        print("... DMA Timeout")
    
    # this is the point to do anything with the data
    # e.g. calculate an FFT of the signal on a CUDA GPU
    print("Calculating FFT...")

    max_value = card.max_sample_value()

    # number of threads in one CUDA block
    num_thread_per_block = 1024

    # copy data to GPU
    data_raw_gpu = cupy.array(data_transfer.buffer)

    # convert raw data to volt
    CupyKernelConvertSignalToVolt = cupy.RawKernel(r'''
        extern "C" __global__
        void CudaKernelScale(const short* anSource, float* afDest, double dFactor) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            afDest[i] = ((float)anSource[i]) * dFactor;
        }
        ''', 'CudaKernelScale')
    data_volt_gpu = cupy.zeros(num_samples, dtype = cupy.float32)
    CupyKernelConvertSignalToVolt((num_samples // num_thread_per_block,), (num_thread_per_block,), (data_raw_gpu, data_volt_gpu, (amplitude_mV / 1000) / max_value))

    # calculate the FFT
    fftdata_gpu = cupy.fft.fft(data_volt_gpu)

    # length of FFT result
    num_fft_samples = num_samples // 2 + 1

    # scale the FFT result
    CupyKernelScaleFFTResult = cupy.RawKernel(r'''
        extern "C" __global__
        void CudaScaleFFTResult (complex<float>* pcompDest, const complex<float>* pcompSource, int lLen) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            pcompDest[i].real (pcompSource[i].real() / (lLen / 2 + 1)); // divide by length of signal
            pcompDest[i].imag (pcompSource[i].imag() / (lLen / 2 + 1)); // divide by length of signal
        }
        ''', 'CudaScaleFFTResult', translate_cucomplex=True)
    CupyKernelScaleFFTResult((num_samples // num_thread_per_block,), (num_thread_per_block,), (fftdata_gpu, fftdata_gpu, num_samples))

    # calculate real spectrum from complex FFT result
    CupyKernelFFTToSpectrum = cupy.RawKernel(r'''
        extern "C" __global__
        void CudaKernelFFTToSpectrum (const complex<float>* pcompSource, float* pfDest) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            pfDest[i] = sqrt (pcompSource[i].real() * pcompSource[i].real() + pcompSource[i].imag() * pcompSource[i].imag());
        }
        ''', 'CudaKernelFFTToSpectrum', translate_cucomplex=True)
    spectrum_gpu = cupy.zeros(num_fft_samples, dtype = cupy.float32)
    CupyKernelFFTToSpectrum((num_fft_samples // num_thread_per_block,), (num_thread_per_block,), (fftdata_gpu, spectrum_gpu))

    # convert to dBFS
    CupyKernelSpectrumToDBFS = cupy.RawKernel(r'''
    extern "C" __global__
    void CudaKernelToDBFS (float* pfDest, const float* pfSource, int lIR_V) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        pfDest[i] = 20. * log10f (pfSource[i] / lIR_V);
    }
    ''', 'CudaKernelToDBFS')
    CupyKernelSpectrumToDBFS((num_fft_samples // num_thread_per_block,), (num_thread_per_block,), (spectrum_gpu, spectrum_gpu, 1))

    spectrum_cpu = cupy.asnumpy(spectrum_gpu)  # copy FFT spectrum back to CPU
    print("done\n")

    # plot FFT spectrum
    plt.figure()
    step = (sample_rate // 2) / (spectrum_cpu.size - 1)
    freq = np.arange(0, sample_rate // 2, step)
    plt.ylim([-140, 0])  # range of Y axis
    plt.plot(freq, spectrum_cpu[:(spectrum_cpu.size - 1)])
    plt.xlabel("Frequency [Hz]")
    plt.show()


