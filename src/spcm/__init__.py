"""
.. include:: ./README.md

.. include:: ./SUPPORTED_DEVICES.md
"""

__docformat__ = "numpy"

# Import all registery entries and spectrum card errors into the module's name space
from .constants import *

# Import all the public classes into the module's namespace
from .classes_device import Device
from .classes_card import Card
from .classes_error_exception import SpcmError, SpcmException, SpcmTimeout
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
from .classes_multi import Multi
from .classes_time_stamp import TimeStamp
from .classes_sequence import Sequence
from .classes_dds import DDS, DDSCore
from .classes_pulse_generators import PulseGenerator, PulseGenerators
from .classes_block_average import BlockAverage
from .classes_boxcar import Boxcar

__all__ = [
    "Device", "Card", "Sync", "CardStack", "Netbox", "CardFunctionality", "Channels", "Channel", "Clock", "Trigger", "MultiPurposeIOs", "MultiPurposeIO",
    "DataTransfer", "DDS", "DDSCore", "PulseGenerator", "PulseGenerators", "Multi", "TimeStamp", "Sequence", "BlockAverage", "Boxcar",
    "SpcmException", "SpcmTimeout", "SpcmError",
]

# Versioning support using versioneer
from . import _version
__version__ = _version.get_versions()['version']

# Writing spcm package version to log file
try:
    from .pyspcm import spcm_dwGetParam_i64, spcm_dwSetParam_ptr, create_string_buffer, byref, int64
    driver_version = int64(0)
    spcm_dwGetParam_i64(None, SPC_GETDRVVERSION, byref(driver_version))
    # Available starting from build 21797
    if (driver_version.value & 0x0000FFFF) < 21797:
        raise OSError("Driver version {} does not support writing spcm version to log".format(driver_version))
    version_str = bytes("Python package spcm v{}".format(__version__), "utf-8")
    version_ptr = create_string_buffer(version_str)
    dwErr = spcm_dwSetParam_ptr(None, SPC_WRITE_TO_LOG, version_ptr, len(version_str))
except OSError as e:
    print(e)

#print(__version__)
