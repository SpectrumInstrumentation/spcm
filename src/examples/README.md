<div style="margin-bottom: 20px; text-align: center">
<a href="https://spectrum-instrumentation.com">
    <img src="https://spectrum-instrumentation.com/img/logo-complete.png"  width=400 />
</a>
</div>

# spcm

A high-level, object-oriented Python package for interfacing with Spectrum Instrumentation GmbH devices.

# Examples

This directory contains `spcm` examples that showcase the different functionalities available in the package and driver. Each sub-folder contains examples of a certain category of functionality.

In these folders, there are several examples with different levels of difficulty. To lower the starting threshold, we've numbered the files. It is recommended to start with the example which name starts with "1_", and after you've understood this example, then continue with the others. 

In addition, to these examples we highly recommend using the corresponding hardware manual for referencing the usage of the API for your specific device.

# Using these examples

To try these examples we recommend the use of Python with a [virtual environment](https://docs.python.org/3/library/venv.html). 

## Download and install Python
Start by installing [Python 3.9 or higher](https://www.python.org/downloads/).

## Create a virtual environment
In the main examples folder, the folder in which this README recides, create a virtual environment:

Under Linux or GitBash (Windows):
```bash
$ python -m venv venv
```

activate the virtual environment and install all the required packages:
```bash
$ source venv/Scripts/activate
$ pip install -r requirements.txt
```

you're all setup to run the examples!

## Running the examples

In the virtual environement, execute the following command to run one of the examples:

```bash
$ python acquisition/1_acq_single.py
```

This will, for example, (on a digitizer) run the example that records a signal and calculates the maximum and minimum and plot the signal with [matplotlib](https://matplotlib.org/).

## Using an IDE to run the examples

After you've created a virtual environment, you can also use typical IDEs for Python (e.g. Visual Studio Code, PyCharm, Eclipse, etc...) to run the examples. Simply open the main examples folder (the folder that contains this README). Please have a look at the specific plugins for the IDEs on how to setup the IDE to use a virtual environment:
* [Visual Studio Code](https://code.visualstudio.com/docs/python/environments)
* [PyCharm](https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html)

From here it's relatively straight forward to getting the examples to run in the IDEs. Simply open one of the examples and press the run button (typically indicated by a play icon).

### Command-Line Interface
Please note that when using for example Git Bash on Windows, it could very well be that you need to put the command `winpty` before the python command to see output and also be able to have keyboard input.

### PyCharm
Please note that when running a script, that the output is shown in the Console. Standardly, this console doesn't take input, hence the `input()` function doesn't work. To get PyCharm to take input in the console you have to check the option `Emulate terminal in output console`, which can be found in the run configuration. It's probably a good idea to have this option always on.

## Structure of an example

Typically, the examples start with:

```python
import spcm

card : spcm.Card
# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(card_type=spcm.SPCM_TYPE_AO) as card:            # if you want to open the first card of a specific type

    # (example code here)
```

With this, the connection to the card is opened and the `card` object can be used to interface with the device.

# See also

* [Spectrum Instrumentation GmbH](https://spectrum-instrumentation.com/)
* [GitHub](https://github.com/SpectrumInstrumentation/spcm)
* [PyPI](https://pypi.org/project/spcm/)
* [Examples](https://github.com/SpectrumInstrumentation/spcm/tree/master/src/examples)
* [Reference API](https://spectruminstrumentation.github.io/spcm/spcm.html)