"""
.. include:: ./README.md

.. include:: ./SUPPORTED_DEVICES.md
"""

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
from .classes_dds import DDS
from .classes_pulse_generators import PulseGenerator, PulseGenerators
from .classes_block_average import BlockAverage
from .classes_boxcar import Boxcar

__all__ = [
    "Device", "Card", "Sync", "CardStack", "Netbox", "CardFunctionality", "Channels", "Channel", "Clock", "Trigger", "MultiPurposeIOs", "MultiPurposeIO",
    "DataTransfer", "DDS", "PulseGenerator", "PulseGenerators", "Multi", "TimeStamp", "Sequence", "BlockAverage", "Boxcar",
    "SpcmException", "SpcmTimeout", "SpcmError",
]

# Versioning support using versioneer
from . import _version
__version__ = _version.get_versions()['version']
#print(__version__)
