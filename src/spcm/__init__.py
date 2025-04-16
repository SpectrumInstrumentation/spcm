"""
.. include:: ./README.md

.. include:: ./SUPPORTED_DEVICES.md
"""

__docformat__ = "numpy"
import numpy as np

# Units available
from pint import UnitRegistry
try:
    import matplotlib.pyplot as plt
    mpl = True
except ImportError:
    mpl = False
units = UnitRegistry(autoconvert_offset_to_baseunit=True)
units.define("sample = 1 = Sa = Sample = Samples = S")
units.define("promille = 0.001 = ‰ = permille = perthousand = perthousands = ppt")
units.define("fraction = 1 = frac = Frac = Fracs = Fraction = Fractions = Frac = Fracs")
units.highZ = np.inf * units.ohm
units.formatter.default_format = "~P" # see https://pint.readthedocs.io/en/stable/user/formatting.html
if mpl:
    units.setup_matplotlib(mpl)
    units.mpl_formatter = "{:~P}" # see https://pint.readthedocs.io/en/stable/user/plotting.html
__all__ = ["units"]

# Import all registery entries and spectrum card errors into the module's name space
from spcm_core import *

# Import all the public classes into the module's namespace
from .classes_device import Device
from .classes_card import Card
from .classes_error_exception import SpcmError, SpcmException, SpcmTimeout, SpcmDeviceNotFound
from .classes_sync import Sync
from .classes_card_stack import CardStack
from .classes_netbox import Netbox
# Functionality
from .classes_functionality import CardFunctionality
from .classes_channels import Channels, Channel
from .classes_clock import Clock
from .classes_trigger import Trigger
from .classes_multi_purpose_ios import MultiPurposeIO, MultiPurposeIOs
from .classes_data_transfer import DataTransfer
from .classes_gated import Gated
from .classes_multi import Multi
from .classes_time_stamp import TimeStamp
from .classes_sequence import Sequence
from .classes_dds import DDS, DDSCore
from .classes_dds_command_list import DDSCommandList
from .classes_dds_command_queue import DDSCommandQueue
from .classes_pulse_generators import PulseGenerator, PulseGenerators
from .classes_aba import ABA
from .classes_block_average import BlockAverage
from .classes_block_statistics import BlockStatistics
from .classes_boxcar import Boxcar
from .classes_scapp import SCAPPTransfer
from .classes_synchronous_digital_ios import SynchronousDigitalIOs

__all__ = [*__all__,
    "Device", "Card", "Sync", "CardStack", "Netbox", "CardFunctionality", "Channels", "Channel", "Clock", "Trigger", "MultiPurposeIOs", "MultiPurposeIO",
    "DataTransfer", "DDS", "DDSCore", "DDSCommandList", "DDSCommandQueue", "PulseGenerator", "PulseGenerators", "Multi", "Gated", "TimeStamp", "Sequence", "ABA",
    "BlockAverage", "Boxcar", "BlockStatistics", "SpcmException", "SpcmTimeout", "SpcmDeviceNotFound", "SpcmError", "SCAPPTransfer", "SynchronousDigitalIOs"
]

# Versioning support using versioneer
from . import _version
__version__ = _version.get_versions()['version']

# Writing spcm package version to log file
try:
    driver_version = int64(0)
    spcm_dwGetParam_i64(None, SPC_GETDRVVERSION, byref(driver_version))
    version_hex = driver_version.value
    major = (version_hex & 0xFF000000) >> 24
    minor = (version_hex & 0x00FF0000) >> 16
    build = version_hex & 0x0000FFFF
    # Available starting from build 21797
    if build < 21797:
        version_str = "v{}.{}.{}".format(major, minor, build)
        raise OSError(f"Driver version build {version_str} does not support writing spcm version to log")
    from importlib.metadata import version
    version_tag = version('spcm')
    version_str = bytes("Python package spcm v{}".format(version_tag), "utf-8")
    version_ptr = create_string_buffer(version_str)
    dwErr = spcm_dwSetParam_ptr(None, SPC_WRITE_TO_LOG, version_ptr, len(version_str))
except OSError as e:
    print(e)
