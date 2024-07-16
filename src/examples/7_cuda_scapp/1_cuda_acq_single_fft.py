"""
Spectrum Instrumentation GmbH (c)

1_cuda_acq_single_fft.py

Shows a simple Standard mode example using only the few necessary commands, with the addition of a CUDA processing step to do an FFT on the GPU
- connect a function generator that generates a sine wave with 1-100 MHz frequency (depending on the max sample rate of your card) and 1 V amplitude to channel 0

Example for analog recording cards (digitizers) for the the M2p, M4i, M4x and M5i card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
from spcm import units

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt


card : spcm.Card

# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(card_type=spcm.SPCM_TYPE_AI) as card:            # if you want to open the first card of a specific type

    # do a simple standard setup
    card.card_mode(spcm.SPC_REC_STD_SINGLE)     # single trigger standard mode
    card.timeout(5 * units.s)

    # setup trigger engine
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)

    # setup channels
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    amplitude = channels[0].amp(1 * units.V, return_unit=units.V)
    amplitude_magnitude_V = amplitude.to(units.V).magnitude
    max_value = card.max_sample_value()

    # we try to use the max samplerate
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)
    sample_rate = clock.sample_rate(max = True, return_unit = units.MHz) # set to maximum sample rate
    print(f"Used Sample Rate: {sample_rate}")

    # setup a data transfer buffer
    num_samples = 64 * units.KiS # KibiSamples = 1024 Samples
    num_samples_magnitude = num_samples.to_base_units().magnitude
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

    # copy data to GPU
    data_raw_gpu = cp.array(data_transfer.buffer)

    # elementwise kernel to convert the raw data to volts
    kernel_signal_to_volt = cp.ElementwiseKernel(
        'T rawData, float64 voltPerLSB', # two inputs: rawData is the integer data (template; can be int8, int16 and int32), voltPerLSB is the factor to convert to volts
        'float32 convertedData', # output is a float32
        'convertedData = rawData * voltPerLSB', # the conversion operation
        'signal_to_volt') # name of the kernel
    
    data_volt_gpu = cp.zeros_like(data_raw_gpu, dtype = cp.float32)
    volt_per_LSB = amplitude_magnitude_V / max_value
    kernel_signal_to_volt(data_raw_gpu, volt_per_LSB, data_volt_gpu)

    # calculate the FFT
    fftdata_gpu = cp.fft.rfft(data_volt_gpu)

    # elementwise kernel to convert the FFT data to a spectrum in dBFS
    kernel_fft_to_spectrum = cp.ElementwiseKernel( 
        'complex64 fftData, int64 numElem, float32 fIR_V', # 3 inputs: complex fft input data; number of samples; input voltage range
        'float32 spectrumData', # output: the spectrum in dBFS
        'spectrumData = 20.0f * log10f ( abs(fftData / thrust::complex<float>(numElem / 2.0f + 1.0f, 0.0f)) / fIR_V)', # the conversion
        'fft_to_spectrum' # name of the conversion
    )

    # scale the FFT result
    spectrum_gpu = cp.empty_like(fftdata_gpu, dtype = cp.float32)
    kernel_fft_to_spectrum(fftdata_gpu, num_samples_magnitude, amplitude_magnitude_V, spectrum_gpu)
    spectrum_cpu = cp.asnumpy(spectrum_gpu)  # copy FFT spectrum back to CPU

    # plot FFT spectrum
    fig, ax = plt.subplots()
    freq = np.fft.rfftfreq(num_samples_magnitude, 1/sample_rate)
    ax.set_ylim([-160, 10])  # range of Y axis
    ax.plot(freq, spectrum_cpu[0,:], ".")
    ax.xaxis.set_units(units.MHz)
    plt.show()


