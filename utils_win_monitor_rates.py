import argparse
import copy
import ctypes
import ctypes.wintypes
import io
import subprocess
import sys

user32 = ctypes.windll.user32

CCHFORMNAME = 32
CCHDEVICENAME = 32
DM_BITSPERPEL = 0x00040000
DM_PELSWIDTH = 0x00080000
DM_PELSHEIGHT = 0x00100000
DM_DISPLAYFLAGS = 0x00200000
DM_DISPLAYFREQUENCY = 0x00400000
DM_POSITION = 0x00000020
DISPLAY_DEVICE_PRIMARY_DEVICE = 0x00000004
ENUM_CURRENT_SETTINGS = -1
CDS_UPDATEREGISTRY = 1
CDS_TEST = 2
CDS_FULLSCREEN = 4
CDS_GLOBAL = 8
CDS_SET_PRIMARY = 16
CDS_RESET = 0x40000000
CDS_SETRECT = 0x20000000
CDS_NORESET = 0x10000000
DISP_CHANGE_SUCCESSFUL = 0
DISP_CHANGE_RESTART = 1
DISP_CHANGE_FAILED = (-1)
DISP_CHANGE_BADMODE = (-2)
DISP_CHANGE_NOTUPDATED = (-3)
DISP_CHANGE_BADFLAGS = (-4)
DISP_CHANGE_BADPARAM = (-5)
DISP_CHANGE_BADDUALVIEW = (-6)


class DUMMYSTRUCT(ctypes.Structure):
    _fields_  = [
        ("dmOrientation", ctypes.c_short),
        ("dmPaperSize", ctypes.c_short),
        ("dmPaperLength", ctypes.c_short),
        ("dmPaperWidth", ctypes.c_short),
        ("dmScale;", ctypes.c_short),
        ("dmCopies;", ctypes.c_short),
        ("dmDefaultSource;", ctypes.c_short),
        ("dmPrintQuality;", ctypes.c_short)
    ]


class DUMMYSTRUCT2(ctypes.Structure):
    _fields_ = [
        ("dmPosition", ctypes.wintypes.POINTL),
        ("dmDisplayOrientation", ctypes.wintypes.DWORD),
        ("dmDisplayFixedOutput", ctypes.wintypes.DWORD)
    ]


class DUMMYUNION(ctypes.Union):
    _anonymous_ = ["s1", "s2"]
    _fields_ = [
        ("s1", DUMMYSTRUCT),
        ("s2", DUMMYSTRUCT2)
    ]


class DUMMYUNION2(ctypes.Union):
    _fields_ = [
        ("dmDisplayFlags", ctypes.wintypes.DWORD),
        ("dmNup", ctypes.wintypes.DWORD)
    ]


class DEVMODE(ctypes.Structure):
    _anonymous_ = ["dummyunion", "dummyunion2"]
    _fields_ = [
        ("dmDeviceName", ctypes.wintypes.BYTE*CCHDEVICENAME),
        ("dmSpecVersion", ctypes.wintypes.WORD),
        ("dmDriverVersion", ctypes.wintypes.WORD),
        ("dmSize", ctypes.wintypes.WORD),
        ("dmDriverExtra", ctypes.wintypes.WORD),
        ("dmFields", ctypes.wintypes.DWORD),
        ("dummyunion", DUMMYUNION),
        ("dmColor", ctypes.c_short),
        ("dmDuplex", ctypes.c_short),
        ("dmYResolution", ctypes.c_short),
        ("dmTTOption", ctypes.c_short),
        ("dmCollate", ctypes.c_short),
        ("dmFormName", ctypes.wintypes.BYTE*CCHFORMNAME),
        ("dmLogPixels", ctypes.wintypes.WORD),
        ("dmBitsPerPel", ctypes.wintypes.DWORD),
        ("dmPelsWidth", ctypes.wintypes.DWORD),
        ("dmPelsHeight", ctypes.wintypes.DWORD),
        ("dummyunion2", DUMMYUNION2),
        ("dmDisplayFrequency", ctypes.wintypes.DWORD),
#if(WINVER >= 0x0400)
        ("dmICMMethod", ctypes.wintypes.DWORD),
        ("dmICMIntent", ctypes.wintypes.DWORD),
        ("dmMediaType", ctypes.wintypes.DWORD),
        ("dmDitherType", ctypes.wintypes.DWORD),
        ("dmReserved1", ctypes.wintypes.DWORD),
        ("dmReserved2", ctypes.wintypes.DWORD),
#if (WINVER >= 0x0500) || (_WIN32_WINNT >= 0x0400)
        ("dmPanningWidth", ctypes.wintypes.DWORD),
        ("dmPanningHeight", ctypes.wintypes.DWORD)
#endif
#endif /* WINVER >= 0x0400 */
    ]


class DISPLAY_DEVICE(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.wintypes.DWORD),
        ("DeviceName", ctypes.wintypes.CHAR*32),
        ("DeviceString", ctypes.wintypes.CHAR*128),
        ("StateFlags", ctypes.wintypes.DWORD),
        ("DeviceID", ctypes.wintypes.CHAR*128),
        ("DeviceKey", ctypes.wintypes.CHAR*128)
    ]


def get_monitor_rates():
    """ Calls to win32 API for returning the connected monitors and their
    refresh rates.

    Returns
    ---------------
    list of tuples(name, rate)
        Returned list of tuples composed of the monitor's name and monitor's
        rate for each connected device.
    """
    monitors = list()

    # Initialize pointers for the adapters
    adapter = DISPLAY_DEVICE()
    adapter.cb = ctypes.sizeof(adapter)
    index = 0
    while user32.EnumDisplayDevicesA(None, index, ctypes.pointer(adapter),0):
        index += 1
        monitor_rate = None
        monitor_name = ""

        # Get the monitor rate
        monitor = DEVMODE()
        monitor.dmSize = ctypes.sizeof(monitor)
        if user32.EnumDisplaySettingsA(
                adapter.DeviceName, ENUM_CURRENT_SETTINGS,
                ctypes.pointer(monitor)
        ):
            monitor_rate = monitor.dmDisplayFrequency

        # Get the monitor name
        j = 0
        while True:
            monitor_info = DISPLAY_DEVICE()
            monitor_info.cb = ctypes.sizeof(monitor_info)
            if not user32.EnumDisplayDevicesA(
                    adapter.DeviceName, j, ctypes.byref(monitor_info), 0):
                break
            monitor_name += monitor_info.DeviceString.decode("utf-8")
            j += 1

        # Store info
        if monitor_rate is not None:
            monitors.append((monitor_name, monitor_rate))

    return monitors

if __name__ == "__main__":
    monitors = get_monitor_rates()
    print(monitors)
