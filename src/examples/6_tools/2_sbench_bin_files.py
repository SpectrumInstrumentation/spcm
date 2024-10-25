"""
Spectrum Instrumentation GmbH (c)

2_sbench_bin_files.py

With this example, you'll be able to import binary files created with SBench. SBench creates binary files in combination with a header file.
The header file contains information about the binary file, such as the number of samples, the sample rate, the resolution, etc. The data
from the binary file is read and plotted.

Example for Netboxes for the the M2p, M4i, M4x and M5i card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import numpy as np
import matplotlib.pyplot as plt

import configparser
import pathlib
path = pathlib.Path(__file__).parent.resolve() / "bin_data"

bin_filename = "export"

# read the configuration file
config = configparser.ConfigParser()
with open(path / (bin_filename + '_binheader.txt')) as hf:
    # Rearrage the header file to be able to read it with the configparser module
    conf_array = hf.read().split("\n\n")
    last_section = conf_array.pop()
    conf_array.insert(1, last_section)
    # Add a root section to simplify the handling
    conf_str = "[root]\n" + "\n".join(conf_array)

    config.read_string(conf_str)

# get the number of channels
num_analog_channels = int(config["root"]["NumAChannels"])
num_digital_channels = int(config["root"]["NumDChannels"])

# get the number of bits
if num_analog_channels:
    resolution = int(config["root"]["Resolution"])
    if resolution <= 8:
        dtype = np.int8
    elif resolution <= 16:
        dtype = np.int16
    else:
        dtype = np.int32
else:
    dtype = np.uint32
    if num_digital_channels > 32:
        dtype = np.uint64

# get the maximum value
max_value = int(config["root"]["MaxADCValue"])

# get the trigger position
trigger_position = int(config["root"]["TrigPosL"]) + (int(config["root"]["TrigPosH"]) << 32)

# get the sample rate
sample_rate = int(config["root"]["Samplerate"])

# get the number of samples
num_samples = int(config["root"]["LenL"]) + (int(config["root"]["LenH"]) << 32)


# read the binary file
with open(path / (bin_filename + '.bin'), 'rb') as f:
    data = np.fromfile(f, dtype=dtype)

if num_analog_channels:
    data = data.reshape((-1, num_analog_channels))
    time_data = np.arange(-trigger_position, num_samples - trigger_position) / sample_rate
    plt.figure()
    for channel in range(num_analog_channels):
        conversion_factor = float(config["Ch{}".format(channel)]["OrigMaxRange"]) / 1000 / max_value
        offset = float(config["Ch{}".format(channel)]["UserOffset"]) / 1e6
        plt.plot(time_data, data[:, channel]*conversion_factor+offset, label="Ch{}".format(channel))
    plt.show()

else:
    def unpackbits(data):
        dshape = list(data.shape)
        return_data = data.reshape([-1, 1])
        num_bits = return_data.dtype.itemsize * 8
        mask = 2**np.arange(num_bits, dtype=return_data.dtype).reshape([1, num_bits])
        return (return_data & mask).astype(bool).astype(int).reshape(dshape + [num_bits])
    
    bit_buffer = unpackbits(data)

    # Plot the acquired data
    time_data = np.arange(-trigger_position, num_samples - trigger_position) / sample_rate
    fig, ax = plt.subplots(bit_buffer.shape[1], 1, sharex=True)
    for channel in range(bit_buffer.shape[1]):
        ax[channel].step(time_data, bit_buffer[:, channel], label=f"{channel}")
        ax[channel].set_ylabel(f"{channel}")
        ax[channel].set_ylim(-0.1, 1.1)
        ax[channel].set_yticks([])
    fig.text(0.04, 0.5, 'channel', va='center', rotation='vertical')
    plt.show()