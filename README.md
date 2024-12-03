<div style="margin-bottom: 20px; text-align: center">
<a href="https://spectrum-instrumentation.com">
    <img src="https://spectrum-instrumentation.com/img/logo-complete.png"  width=400 />
</a>
</div>

# spcm
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI - Version](https://img.shields.io/pypi/v/spcm)](https://pypi.org/project/spcm)
[![PyPi Downloads](https://img.shields.io/pypi/dm/spcm?label=downloads%20%7C%20pip&logo=PyPI)](https://pypi.org/project/spcm)
[![Follow](https://img.shields.io/twitter/follow/SpecInstruments.svg?style=social&style=flat&logo=twitter&label=Follow&color=blue)](https://twitter.com/SpecInstruments/)
![GitHub followers](https://img.shields.io/github/followers/SpectrumInstrumentation)

A high-level, object-oriented Python package for interfacing with Spectrum Instrumentation GmbH devices.

`spcm` can handle individual cards (`Card`), StarHubs (`Sync`), groups of cards (`CardStack`) and Netboxes (`Netbox`).

# Supported devices

See the [SUPPORTED_DEVICES.md](https://github.com/SpectrumInstrumentation/spcm/blob/master/src/spcm/SUPPORTED_DEVICES.md) file for a list of supported devices.

# Requirements
[![Static Badge](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/)
[![Static Badge](https://img.shields.io/badge/NumPy-1.25+-green)](https://numpy.org/)
[![Static Badge](https://img.shields.io/badge/pint-0.24+-teal)](https://pint.readthedocs.io/en/stable/)

`spcm` requires the Spectrum Instrumentation [driver](https://spectrum-instrumentation.com/support/downloads.php) which is available for Windows and Linux. 
Please have a look in the manual of your product for more information about installing the driver on the different plattforms.

## Optional Requirements
[![Static Badge](https://img.shields.io/badge/cuda--python-12.6+-green)](https://developer.nvidia.com/cuda-python)
[![Static Badge](https://img.shields.io/badge/cupy--cuda12x-13.3+-green)](https://cupy.dev/)
[![Static Badge](https://img.shields.io/badge/h5py-3.10+-orange)](https://www.h5py.org/)

These are dependencies that the `spcm` package uses, when they are installed. See the `src/examples` folder for specific use cases.

# Installation and dependencies
[![Pip Package](https://img.shields.io/pypi/v/spcm?logo=PyPI)](https://pypi.org/project/spcm)
[![Publish to PyPI](https://github.com/SpectrumInstrumentation/spcm/actions/workflows/spcm-publish-to-pypi.yml/badge.svg)](https://github.com/SpectrumInstrumentation/spcm/actions/workflows/spcm-publish-to-pypi.yml)

Start by installing Python 3.9 or higher. We recommend using the latest version. You can download Python from [https://www.python.org/](https://www.python.org/).

You would probably also like to install and use a virtual environment, although it's not strictly necessary. See the examples [README.md](https://github.com/SpectrumInstrumentation/spcm/blob/master/src/examples/README.md) for a more detailed explanation on how to use `spcm` in a virtual environment.

To install the latest release using `pip`:
```bash
$ pip install spcm
```
Note that: this will automatically install all the dependencies.

# Documentation
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://spectruminstrumentation.github.io/spcm/spcm.html)
[![Build dosc](https://github.com/SpectrumInstrumentation/spcm/actions/workflows/spcm-docs-pages.yml/badge.svg)](https://github.com/SpectrumInstrumentation/spcm/actions/workflows/spcm-docs-pages.yml)
[![Publish docs](https://github.com/SpectrumInstrumentation/spcm/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/SpectrumInstrumentation/spcm/actions/workflows/pages/pages-build-deployment)

The API documentation for the latest [stable release](https://spectruminstrumentation.github.io/spcm/spcm.html) is available for reading on GitHub pages.

Please also see the hardware user manual for your specific card for more information about the available functionality.

# Using spcm

The `spcm` package is a high-level object-oriented programming library for controlling Spectrum Instrumentation devices.

## Examples
For detailed examples see the `src\examples` directory. There are several sub-directories each corresponding to a certain kind of functionality. You can find the most recent examples on [GitHub](https://github.com/SpectrumInstrumentation/spcm/tree/master/src/examples).


## Hardware interfaces

`spcm` provides the following classes for interfacing with the different devices.

| Name        | Description                                                                       |
|-------------|-----------------------------------------------------------------------------------|
| `Card`      | a class to control the low-level API of Spectrum Instrumentation cards. |
| `Sync`      | a class for controling StarHub devices.                  |
| `CardStack` | a class that handles the opening and closing of a combination of different cards either with or without a StarHub that synchronizes the cards. |
| `Netbox`    | a class that handles the opening and closing of the cards in a Netbox                          |


## Connect to a device
Opening and closing of cards is handled using the python [`with`](https://peps.python.org/pep-0343/) statement. This creates a context manager that safely handles opening and closing of a card or a group of cards.

### Using device identifiers
Connect to local cards:

```python
import spcm

with spcm.Card('/dev/spcm0') as card:

    # (add your code here)
```
Connect to remote cards (you can find a card's IP using the
[Spectrum Control Center](https://spectrum-instrumentation.com/en/spectrum-control-center) software):

```python
import spcm

with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:
    
    # (add your code here)
```

Connect to a group of cards synchronized using a StarHub:

```python
import spcm

card_identifiers = ["/dev/spcm0", "/dev/spcm1"]
sync_identifier  = "sync0"

with spcm.CardStack(card_identifiers=card_identifiers, sync_identifier=sync_identifier) as stack:

    # (add your code here)
```
The `CardStack` object contains a list of `Card` objects in the `stack.cards` parameter and a `Sync` object in the parameter `stack.sync`.

### Using card type or serial number

Apart from connecting to a device directly through a device identifier it's also possible to connect to local devices using the card type or serial number. 

To find the first card of type analog out (`SPCM_TYPE_AO`) you can do the following:
```python
import spcm

with spcm.Card(card_type=spcm.SPCM_TYPE_AO) as card:
    
    # (add your code here)
```
See the register `SPC_FNCTYPE` in the reference manual of your specific device for the kind of card you're using.

If you want to connect to a device based on it's serial number, do the following:
```python
import spcm

with spcm.Card(serial_number=[your serial number here]) as card:
    
    # (add your code here)
```
See the register `SPC_PCISERIALNO` in the reference manual of your specific device for more information.

If the `device_identifier` is given, that card is opened, if at the same time `card_type` or `serial_number` are given, then these behave as an additional check too see if the opened card is of a certain type or has that specific serial number.

### Demo devices
To test the Spectrum Instrumentation API with user code without hardware, the Control Center gives the user the option to create [demo devices](https://spectrum-instrumentation.com/support/knowledgebase/software/How_to_set_up_a_demo_card.php). These demo devices can be used in the same manner as real devices. Simply change the device identifier string to the string as shown in the Control Center.

## Units and quantities

`spcm` uses [pint](https://pint.readthedocs.io/en/stable/) to handle all quantities that have a physical unit. To enable the use of quantities simply import the units module from `spcm`:

```python
from spcm import units
```

This imports the `units` object from `spcm`. This is a `UnitRegistry` object from `pint`. Defining a quantity is as simple as:

```python
from spcm import units

frequency = 100 * units.MHz
amplitude = 1 * units.V
# etc...
```

All methods within the `spcm` that expect a value related to a physical quantity, support these quantities, for example setting a timeout:

```python
card.timeout(5 * units.s)
```

See our dedicated examples for more information about where units can be used.

## Card Functionality
After opening a card, StarHub, group of cards or Netbox, specific functionality of the cards can be accessed through `CardFunctionality` classes. 

| Name                | Description                                                         |
|---------------------|---------------------------------------------------------------------|
| `Channels`          | class for setting up the in- or output stage of the channels of a card |
| `Clock`             | class for setting up the clock engine of the card                   |
| `Trigger`           | class for setting up the trigger engine of the card                 |
| `MultiPurposeIOs`   | class for setting up the multi purpose i/o of the card              |
| `DataTransfer`      | class for handling data transfer functionality                      |
| `Multi`             | class for handling multiple recording and replay mode functionality |
| `Sequence`          | class for handling sequence mode functionality                      |
| `TimeStamp`         | class for handling time stamped data                                |
| `Boxcar`            | class for handling boxcar averaging                                 |
| `BlockAverage`      | class for handling block averaging functionality                    |
| `PulseGenerators`   | class for setting up the pulse generator functionality              |
| `DDS`               | class for handling DDS functionality                                |
| `DDSCommandList`    | class for abstracting streaming DDS commands using blocks of commands |
| `DDSCommandQueue`   | class for abstracting streaming DDS commands using a queue of commands |

To use a specific functionality simply initiate an instance of one of the classes and pass a device object:

```python
import spcm

with spcm.Card('/dev/spcm0') as card:

    clock = spcm.Clock(card)
    # (or)
    trigger = spcm.Trigger(card)
    # (or)
    multi_io = spcm.MultiPurposeIOs(card)
    # (or)
    data_transfer = spcm.DataTransfer(card)
    # etc ...

```
Each of these functionalities typically corresponds to a chapter in your device manual, so for further reference please have a look in the device manual.

## Setting up Channels

The Channels functionality allows the user to setup individual channels on a Card or a CardStack. The channels object is also a Python iterator, hence if a channels object is used it a for-loop, each iteration provides a Channel object:

```python
channels = spcm.Channels(card, card_enable=spcm.CHANNEL0 | spcm.CHANNEL1)
for channel in channels:
    # do something with each channel
```

In addition, the user can define the output load connected to the channel (standard value 50 Ohm), to any resistor value or high impedance (`units.highZ`). With this output load, the amplitude setting is done with repect to this output load:

```python
channels = Channels(card, card_enable=spcm.CHANNEL0 | spcm.CHANNEL1)
channels[0].output_load(units.highZ)
channels[0].amp(1 * units.V)
# or for all channels
channels.output_load(units.highZ)
channels.amp(1 * units.V)
```

This information allows the user to convert the data coming from `DataTransfer`, using the `convert_data()` method. See the section "Setting up a data transfer buffer ..." for more details.

## Setting up the Clock engine

The Clock engine is used to generate a clock signal that is used as the source for all timing critical processes on the card.

### Sample rate
To get the maximum sample rate of the active card and set the sample rate to the maximum, this is an example using internal PLL clock mode:
```python
clock = spcm.Clock(card)
clock.mode(spcm.SPC_CM_INTPLL)
sample_rate = clock.sample_rate(max=True) 
# (or) 
sample_rate = clock.sample_rate(20 * units.MHz) # for a 20 MHz sample rate (see reference manual for allowed values)
print("Current sample rate: {}".format(sample_rate, return_unit=units.MS/units.s))
```

## Setting up the Trigger engine

### External trigger

The Trigger engine can be configured for a multitude of different configurations (see the hardware manual for more information about the specific configurations for your device). Here we've given an example for an external trigger arriving at input port ext0, that is DC-coupled. The card is waiting for positiv edge that excedes 1.5 V:

```python
trigger = spcm.Trigger(card)
trigger.or_mask(spcm.SPC_TMASK_EXT0) # set the ext0 hardware input as trigger source
trigger.ext0_mode(spcm.SPC_TM_POS) # wait for a positive edge
trigger.ext0_level0(1.5 * units.V)
trigger.ext0_coupling(spcm.COUPLING_DC) # set DC coupling
```

## Setting up the multi-purpose I/O lines

See the hardware manual, for the multi-purpose I/O lines functionality that can be programmed.

## Setting up a data transfer buffer for recording (digitizer) or replay (AWG)

### Recording (Digitizing)
To transfer data to or from the card, we have to setup a data transfer object. This object allocates an amount the memory of the card (`memory_size`) and a Direct Memory Access (DMA) buffer on the host pc (`allocate_buffer`). Half of the samples are taken before the trigger, as configured by the trigger engine, and half of the samples are recorded afterwards. Then the transfer from the card to the host pc is started and the program waits until the DMA is filled.

```python
# define the data buffer
num_samples = 4 * units.KiS # 1 KiS = 1 KibiSamples = 1024 Samples
data_transfer = spcm.DataTransfer(card)
data_transfer.memory_size(num_samples)
data_transfer.allocate_buffer(num_samples)
data_transfer.post_trigger(num_samples // 2)
# Start DMA transfer
data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)

# start card and wait until the memory is filled
card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_DATA_WAITDMA)

# (your code handling the recorded data comes here)
```

To access the recorded data, the data transfer object holds a [NumPy](https://numpy.org/) array object `data_transfer.buffer` that is directly mapped to the DMA buffer. This buffer can be directly accessed by the different analysis methods available in the NumPy package:

```python
import numpy as np

# The extrema of the data
minimum = np.min(data_transfer.buffer, axis=1)
maximum = np.max(data_transfer.buffer, axis=1)
```

This data can be further processed or plotted using, for example, [`matplotlib`](https://matplotlib.org/).

### Replay (Generation)

The setup of the DMA for replay is very similar. First card memory is allocated with `memory_size` and then a DMA buffer is allocated (`allocated_buffer`) and made accessible though the NumPy object `data_transfer.buffer`, which can then be written to using standard NumPy methods. Finally, data transfer from the host PC to the card is started and the programming is waiting until all the data is transferred:
```python
num_samples = 4 * units.KiS # 1 KiS = 1 KibiSamples = 1024 Samples
data_transfer = spcm.DataTransfer(card)
data_transfer.memory_size(num_samples)
data_transfer.allocate_buffer(num_samples)
data_transfer.loops(0) # loop continuously
# simple linear ramp for analog output cards
num_samples_magnitude = num_samples.to(units.S).magnitude
data_transfer.buffer[:] = np.arange(-num_samples_magnitude//2, num_samples_magnitude//2).astype(np.int16)

data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA, spcm.M2CMD_DATA_WAITDMA)
```

#### Data transfer

In the above it's assume that all the data has been transferred from the card into the buffer object, however if you're working in FIFO mode or you've defined the buffer to be smaller then the assigned card memory size, then the card will write to parts of the memory and tell you were it wrote. 

This can be either handled manually (see the reference manual of your card, or the methods `avail_card_len()`, `avail_user_pos()` and `avail_user_len()`).

Secondly, it can also be handled with Python iterators functionality. If the `data_transfer` object is handed to a for-loop, then in each iteration a view of the current active memory part is given in the form of a NumPy array and the user can do calculations on that.

```python
# Get a block of data
for data_block in data_transfer:
    # data_block is a NumPy array view of the currently active buffer part
    for channel in channels:
        data_units = channel.convert_data(data_block[:, channel])
        minimum = np.min(data_units)
        maximum = np.max(data_units)
        print(f"Minimum: {minimum} - maximum: {maximum}")
```

The same principle also works with generator cards (see the example `2_gen_fifo.py`).

## Multiple recording / replay

In case of multiple recording / replay, the memory is divided in equally sized segments, that are populated / replayed when a trigger is detected. Multiple triggers each trigger the next segment to be populated or replayed.

The following code snippet shows how to setup the buffer for 4 segments with each 128 Samples:

```python
samples_per_segment = 128 * units.S
num_segments = 4
multiple_recording = spcm.Multi(card)
multiple_recording.memory_size(samples_per_segment*num_segments)
multiple_recording.allocate_buffer(segment_samples=samples_per_segment, num_segments=num_segments)
multiple_recording.post_trigger(samples_per_segment // 2)
multiple_recording.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)
```
Again there are half the samples before and half the samples after the trigger.

## Timestamps

See the example `6_acq_fifo_multi_ts_poll.py` in the examples folder `1_acquisition` for more information about the usage of timestamps. Moreover, detailed information about timestamps can be found in the corresponding chapter in the specific hardware manual.

To setup the timestamp buffer:

```python
ts = spcm.TimeStamp(card)
ts.mode(spcm.SPC_TSMODE_STARTRESET, spcm.SPC_TSCNT_INTERNAL)
ts.allocate_buffer(num_time_stamps)
```

The user can then use polling to acquire time stamp data from the card.

## Additional functionality

### SCAPP with CuPy (add-on option)

For more information about the SCAPP option, please see the detailed information page: https://spectrum-instrumentation.com/products/drivers_examples/scapp_cuda_interface.php.

Please see the folder `7_cuda_scapp` in the `examples` folder for several dedicated exampes.

### Pulse generator

Please see the folder `4_pulse-generator` in the `examples` folder for several dedicated examples. In the following, there is a simple example for setting up a single pulse generator on x0.

Create a pulse generators object and get the clock rate used by the pulse generator. Use that to calculate the period of a 1 MHz signal and the half of that period we'll have a high signal (hence 50% duty cycle). The pulse generator will start if the trigger condition is met without delay and loops infinitely many times. The triggering condition is set to the card software trigger. See more details in the pulse generator chapter in the specific hardware manual.

```python
pulse_generators = spcm.PulseGenerators(card)

# generate a continuous signal with 1 MHz
pulse_generators[0].pulse_period(1 * units.us)
# pulse_generators[0].repetition_rate(1 * units.MHz) # (or)
pulse_generators[0].duty_cycle(50 * units.percent)
# pulse_generators[0].pulse_length(0.5 * units.us) # (or)
pulse_generators[0].start_delay(0 * units.us)
pulse_generators[0].repetitions(0) # 0: infinite
pulse_generators[0].start_condition_state_signal(spcm.SPCM_PULSEGEN_MUX1_SRC_UNUSED)
pulse_generators[0].start_condition_trigger_signal(spcm.SPCM_PULSEGEN_MUX2_SRC_SOFTWARE)
pulse_generators[0].invert_start_condition(False)

# and write the pulse generator settings
pulse_generators.write_setup()

# start all pulse generators that wait for a software command
pulse_generators.force()
```

This will start the pulse generator to continuously output a 1 MHz signal.

### Sequence replay mode (AWG only)

Please see the example `3_gen_sequence.py` in the folder `2_generation` of the examples to see an example dedicated to the Sequence replay mode.

### DDS (AWG only)

Please see the examples in the dedicated examples folder `3_dds` for all the functionality provided by the DDS framework. Moreover, please also have a look at the corresponding hardware manual.

### Boxcar (Digitizer only)

See the corresponding chapter in the hardware manual for more information about boxcar averaging and the registers used.

### Block average (Digitizer only)

See the corresponding chapter in the hardware manual for more information about block averaging and the registers used.

# Acknowledgements

We would like to thank [Christian Baker](https://github.com/crnbaker) for fruitful discussions and inspiration on how to setup the `spcm` package.