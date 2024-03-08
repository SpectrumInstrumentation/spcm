import os
import platform
import sys
from ctypes import *

# load registers for easier access
from .regs import *

# load registers for easier access
from .spcerr import *

SPCM_DIR_PCTOCARD = 0
SPCM_DIR_CARDTOPC = 1

SPCM_BUF_DATA      = 1000 # main data buffer for acquired or generated samples
SPCM_BUF_ABA       = 2000 # buffer for ABA data, holds the A-DATA (slow samples)
SPCM_BUF_TIMESTAMP = 3000 # buffer for timestamps

TYPE_INT64 = 0
TYPE_DOUBLE = 1

# determine bit width of os
oPlatform = platform.architecture()
if (oPlatform[0] == '64bit'):
    bIs64Bit = 1
else:
    bIs64Bit = 0

# define pointer aliases
int8  = c_int8
int16 = c_int16
int32 = c_int32
int64 = c_int64

ptr8  = POINTER (int8)
ptr16 = POINTER (int16)
ptr32 = POINTER (int32)
ptr64 = POINTER (int64)

uint8  = c_uint8
uint16 = c_uint16
uint32 = c_uint32
uint64 = c_uint64

uptr8  = POINTER (uint8)
uptr16 = POINTER (uint16)
uptr32 = POINTER (uint32)
uptr64 = POINTER (uint64)

double = c_double
dptr64 = POINTER (double)

class _U(Union):
    """Union for doubles and 64-bit integers"""
    _fields_ = [
        ("dValue", double),
        ("llValue", int64)
    ]

class ST_LIST_PARAM(Structure):
    """Structure for lists of parameters"""
    _anonymous_ = ("Value",)
    _fields_ = [
        ("lReg", int32), # the register
        ("lType", int32), # the type of value written
        ("Value", _U), # the actual value
    ]

# Check for driver versions
try:
    # Windows
    if os.name == 'nt':
        # define card handle type
        if (bIs64Bit):
            # for unknown reasons c_void_p gets messed up on Win7/64bit, but this works:
            drv_handle = POINTER(c_uint64)
        else:
            drv_handle = c_void_p

        # Load DLL into memory.
        # use windll because all driver access functions use _stdcall calling convention under windows
        if (bIs64Bit == 1):
            spcmDll = windll.LoadLibrary ("spcm_win64.dll")
        else:
            spcmDll = windll.LoadLibrary ("spcm_win32.dll")

        # load spcm_hOpen
        if (bIs64Bit):
            spcm_hOpen = getattr(spcmDll, "spcm_hOpen")
        else:
            spcm_hOpen = getattr(spcmDll, "_spcm_hOpen@4")
        spcm_hOpen.argtype = [c_char_p]
        spcm_hOpen.restype = drv_handle

        # load spcm_vClose
        if (bIs64Bit):
            spcm_vClose = getattr(spcmDll, "spcm_vClose")
        else:
            spcm_vClose = getattr(spcmDll, "_spcm_vClose@4")
        spcm_vClose.argtype = [drv_handle]
        spcm_vClose.restype = None

        # load spcm_dwGetErrorInfo
        if (bIs64Bit):
            spcm_dwGetErrorInfo_i32 = getattr(spcmDll, "spcm_dwGetErrorInfo_i32")
        else:
            spcm_dwGetErrorInfo_i32 = getattr(spcmDll, "_spcm_dwGetErrorInfo_i32@16")
        spcm_dwGetErrorInfo_i32.argtype = [drv_handle, uptr32, ptr32, c_char_p]
        spcm_dwGetErrorInfo_i32.restype = uint32

        # load spcm_dwGetParam_i32
        if (bIs64Bit):
            spcm_dwGetParam_i32 = getattr(spcmDll, "spcm_dwGetParam_i32")
        else:
            spcm_dwGetParam_i32 = getattr(spcmDll, "_spcm_dwGetParam_i32@12")
        spcm_dwGetParam_i32.argtype = [drv_handle, int32, ptr32]
        spcm_dwGetParam_i32.restype = uint32

        # load spcm_dwGetParam_i64
        if (bIs64Bit):
            spcm_dwGetParam_i64 = getattr(spcmDll, "spcm_dwGetParam_i64")
        else:
            spcm_dwGetParam_i64 = getattr(spcmDll, "_spcm_dwGetParam_i64@12")
        spcm_dwGetParam_i64.argtype = [drv_handle, int32, ptr64]
        spcm_dwGetParam_i64.restype = uint32

        # load spcm_dwGetParam_d64
        if (bIs64Bit):
            spcm_dwGetParam_d64 = getattr(spcmDll, "spcm_dwGetParam_d64")
        else:
            spcm_dwGetParam_d64 = getattr(spcmDll, "_spcm_dwGetParam_d64@12")
        spcm_dwGetParam_d64.argtype = [drv_handle, int32, dptr64]
        spcm_dwGetParam_d64.restype = uint32

        # load spcm_dwGetParam_ptr
        if (bIs64Bit):
            spcm_dwGetParam_ptr = getattr(spcmDll, "spcm_dwGetParam_ptr")
        else:
            spcm_dwGetParam_ptr = getattr(spcmDll, "_spcm_dwGetParam_ptr@12")
        spcm_dwGetParam_ptr.argtype = [drv_handle, int32, c_void_p, uint64]
        spcm_dwGetParam_ptr.restype = uint32

        # load spcm_dwSetParam_i32
        if (bIs64Bit):
            spcm_dwSetParam_i32 = getattr(spcmDll, "spcm_dwSetParam_i32")
        else:
            spcm_dwSetParam_i32 = getattr(spcmDll, "_spcm_dwSetParam_i32@12")
        spcm_dwSetParam_i32.argtype = [drv_handle, int32, int32]
        spcm_dwSetParam_i32.restype = uint32

        # load spcm_dwSetParam_i64
        if (bIs64Bit):
            spcm_dwSetParam_i64_ = getattr(spcmDll, "spcm_dwSetParam_i64")
        else:
            spcm_dwSetParam_i64_ = getattr(spcmDll, "_spcm_dwSetParam_i64@16")
        spcm_dwSetParam_i64_.argtype = [drv_handle, int32, int64]
        spcm_dwSetParam_i64_.restype = uint32

        # load spcm_dwSetParam_d64
        if (bIs64Bit):
            spcm_dwSetParam_d64_ = getattr(spcmDll, "spcm_dwSetParam_d64")
        else:
            spcm_dwSetParam_d64_ = getattr(spcmDll, "_spcm_dwSetParam_d64@16")
        spcm_dwSetParam_d64_.argtype = [drv_handle, int32, double]
        spcm_dwSetParam_d64_.restype = uint32

        # load spcm_dwSetParam_ptr
        if (bIs64Bit):
            spcm_dwSetParam_ptr = getattr(spcmDll, "spcm_dwSetParam_ptr")
        else:
            spcm_dwSetParam_ptr = getattr(spcmDll, "_spcm_dwSetParam_ptr@16")
        spcm_dwGetParam_ptr.argtype = [drv_handle, int32, c_void_p, uint64]
        spcm_dwSetParam_ptr.restype = uint32

        # load spcm_dwSetParam_i64m
        if (bIs64Bit):
            spcm_dwSetParam_i64m = getattr(spcmDll, "spcm_dwSetParam_i64m")
        else:
            spcm_dwSetParam_i64m = getattr(spcmDll, "_spcm_dwSetParam_i64m@16")
        spcm_dwSetParam_i64m.argtype = [drv_handle, int32, int32, int32]
        spcm_dwSetParam_i64m.restype = uint32

        # load spcm_dwDefTransfer_i64
        if (bIs64Bit):
            spcm_dwDefTransfer_i64_ = getattr(spcmDll, "spcm_dwDefTransfer_i64")
        else:
            spcm_dwDefTransfer_i64_ = getattr(spcmDll, "_spcm_dwDefTransfer_i64@36")
        spcm_dwDefTransfer_i64_.argtype = [drv_handle, uint32, uint32, uint32, c_void_p, uint64, uint64]
        spcm_dwDefTransfer_i64_.restype = uint32

        # load spcm_dwInvalidateBuf
        if (bIs64Bit):
            spcm_dwInvalidateBuf = getattr(spcmDll, "spcm_dwInvalidateBuf")
        else:
            spcm_dwInvalidateBuf = getattr(spcmDll, "_spcm_dwInvalidateBuf@8")
        spcm_dwInvalidateBuf.argtype = [drv_handle, uint32]
        spcm_dwInvalidateBuf.restype = uint32

        # load spcm_dwGetContBuf_i64
        if (bIs64Bit):
            spcm_dwGetContBuf_i64 = getattr(spcmDll, "spcm_dwGetContBuf_i64")
        else:
            spcm_dwGetContBuf_i64 = getattr(spcmDll, "_spcm_dwGetContBuf_i64@16")
        spcm_dwGetContBuf_i64.argtype = [drv_handle, uint32, POINTER(c_void_p), uptr64]
        spcm_dwGetContBuf_i64.restype = uint32

        # load spcm_dwDiscovery
        if (bIs64Bit):
            spcm_dwDiscovery = getattr(spcmDll, "spcm_dwDiscovery")
        else:
            spcm_dwDiscovery = getattr(spcmDll, "_spcm_dwDiscovery@16")
        spcm_dwDiscovery.argtype = [POINTER(c_char_p), uint32, uint32, uint32]
        spcm_dwDiscovery.restype = uint32

        # load spcm_dwSendIDNRequest
        if (bIs64Bit):
            spcm_dwSendIDNRequest = getattr(spcmDll, "spcm_dwSendIDNRequest")
        else:
            spcm_dwSendIDNRequest = getattr(spcmDll, "_spcm_dwSendIDNRequest@12")
        spcm_dwSendIDNRequest.argtype = [POINTER(c_char_p), uint32, uint32]
        spcm_dwSendIDNRequest.restype = uint32


    elif os.name == 'posix':
        sys.stdout.write("Python Version: {0} on Linux\n\n".format (platform.python_version()))

        # define card handle type
        if (bIs64Bit):
            drv_handle = POINTER(c_uint64)
        else:
            drv_handle = c_void_p

        # Load DLL into memory.
        # use cdll because all driver access functions use cdecl calling convention under linux
        spcmSo = cdll.LoadLibrary ("libspcm_linux.so")

        # load spcm_hOpen
        spcm_hOpen = getattr(spcmSo, "spcm_hOpen")
        spcm_hOpen.argtype = [c_char_p]
        spcm_hOpen.restype = drv_handle

        # load spcm_vClose
        spcm_vClose = getattr(spcmSo, "spcm_vClose")
        spcm_vClose.argtype = [drv_handle]
        spcm_vClose.restype = None

        # load spcm_dwGetErrorInfo
        spcm_dwGetErrorInfo_i32 = getattr(spcmSo, "spcm_dwGetErrorInfo_i32")
        spcm_dwGetErrorInfo_i32.argtype = [drv_handle, uptr32, ptr32, c_char_p]
        spcm_dwGetErrorInfo_i32.restype = uint32

        # load spcm_dwGetParam_i32
        spcm_dwGetParam_i32 = getattr(spcmSo, "spcm_dwGetParam_i32")
        spcm_dwGetParam_i32.argtype = [drv_handle, int32, ptr32]
        spcm_dwGetParam_i32.restype = uint32

        # load spcm_dwGetParam_i64
        spcm_dwGetParam_i64 = getattr(spcmSo, "spcm_dwGetParam_i64")
        spcm_dwGetParam_i64.argtype = [drv_handle, int32, ptr64]
        spcm_dwGetParam_i64.restype = uint32

        # load spcm_dwGetParam_d64
        spcm_dwGetParam_d64 = getattr(spcmSo, "spcm_dwGetParam_d64")
        spcm_dwGetParam_d64.argtype = [drv_handle, int32, dptr64]
        spcm_dwGetParam_d64.restype = uint32

        # load spcm_dwGetParam_ptr
        spcm_dwGetParam_ptr = getattr(spcmSo, "spcm_dwGetParam_ptr")
        spcm_dwGetParam_ptr.argtype = [drv_handle, int32, c_void_p, uint64]
        spcm_dwGetParam_ptr.restype = uint32

        # load spcm_dwSetParam_i32
        spcm_dwSetParam_i32 = getattr(spcmSo, "spcm_dwSetParam_i32")
        spcm_dwSetParam_i32.argtype = [drv_handle, int32, int32]
        spcm_dwSetParam_i32.restype = uint32

        # load spcm_dwSetParam_i64
        spcm_dwSetParam_i64_ = getattr(spcmSo, "spcm_dwSetParam_i64")
        spcm_dwSetParam_i64_.argtype = [drv_handle, int32, int64]
        spcm_dwSetParam_i64_.restype = uint32

        # load spcm_dwSetParam_i64m
        spcm_dwSetParam_i64m = getattr(spcmSo, "spcm_dwSetParam_i64m")
        spcm_dwSetParam_i64m.argtype = [drv_handle, int32, int32, int32]
        spcm_dwSetParam_i64m.restype = uint32

        # load spcm_dwSetParam_d64
        spcm_dwSetParam_d64_ = getattr(spcmSo, "spcm_dwSetParam_d64")
        spcm_dwSetParam_d64_.argtype = [drv_handle, int32, double]
        spcm_dwSetParam_d64_.restype = uint32

        # load spcm_dwSetParam_ptr
        spcm_dwSetParam_ptr = getattr(spcmSo, "spcm_dwSetParam_ptr")
        spcm_dwSetParam_ptr.argtype = [drv_handle, int32, c_void_p, uint64]
        spcm_dwSetParam_ptr.restype = uint32

        # load spcm_dwDefTransfer_i64
        spcm_dwDefTransfer_i64_ = getattr(spcmSo, "spcm_dwDefTransfer_i64")
        spcm_dwDefTransfer_i64_.argtype = [drv_handle, uint32, uint32, uint32, c_void_p, uint64, uint64]
        spcm_dwDefTransfer_i64_.restype = uint32

        # load spcm_dwInvalidateBuf
        spcm_dwInvalidateBuf = getattr(spcmSo, "spcm_dwInvalidateBuf")
        spcm_dwInvalidateBuf.argtype = [drv_handle, uint32]
        spcm_dwInvalidateBuf.restype = uint32

        # load spcm_dwGetContBuf_i64
        spcm_dwGetContBuf_i64 = getattr(spcmSo, "spcm_dwGetContBuf_i64")
        spcm_dwGetContBuf_i64.argtype = [drv_handle, uint32, POINTER(c_void_p), uptr64]
        spcm_dwGetContBuf_i64.restype = uint32

        # load spcm_dwDiscovery
        spcm_dwDiscovery = getattr(spcmSo, "spcm_dwDiscovery")
        spcm_dwDiscovery.argtype = [POINTER(c_char_p), uint32, uint32, uint32]
        spcm_dwDiscovery.restype = uint32

        # load spcm_dwSendIDNRequest
        spcm_dwSendIDNRequest = getattr(spcmSo, "spcm_dwSendIDNRequest")
        spcm_dwSendIDNRequest.argtype = [POINTER(c_char_p), uint32, uint32]
        spcm_dwSendIDNRequest.restype = uint32

    else:
        raise Exception('Operating system not supported')

except OSError as e:
    raise Exception(text="The Spectrum Instrumentation device driver is not found. Please install the driver and try again.\nFor the newest drivers, see https://spectrum-instrumentation.com/support/downloads.php")

except AttributeError as e:
    minimum_driver_version = "7.0"
    raise Exception(text="Driver version not supported. Minimum version required: {}.\n For the newest drivers, see https://spectrum-instrumentation.com/support/downloads.php".format(minimum_driver_version))

def spcm_dwSetParam_i64(hDrv, lReg, Val):
    try:
        llVal = int64(Val.value)
    except AttributeError:
        llVal = int64(Val)
    return spcm_dwSetParam_i64_ (hDrv, lReg, llVal)

def spcm_dwSetParam_d64(hDrv, lReg, Val):
    try:
        dVal = double(Val.value)
    except AttributeError:
        dVal = double(Val)
    return spcm_dwSetParam_d64_ (hDrv, lReg, dVal)


def spcm_dwDefTransfer_i64(hDrv, dwBufferType, dwDirection, dwNotify, pvBuffer, offs, buf_len):
    try:
        qwOffs = uint64(offs.value)
    except AttributeError:
        qwOffs = uint64(offs)
    try:
        qwBufLen = uint64(buf_len.value)
    except AttributeError:
        qwBufLen = uint64(buf_len)
    return spcm_dwDefTransfer_i64_ (hDrv, dwBufferType, dwDirection, dwNotify, pvBuffer, qwOffs, qwBufLen)
