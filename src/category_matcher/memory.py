"""Read this process's native memory counters without inspecting other processes."""

import ctypes
import sys


class TaskVMInfoPrefix(ctypes.Structure):
    # Apple's SDK mach/task_info.h uses pack(4). TASK_VM_INFO rev1 ends at
    # phys_footprint; later kernels may return additional fields after it.
    _pack_ = 4
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("region_count", ctypes.c_int32),
        ("page_size", ctypes.c_int32),
        *[
            (name, ctypes.c_uint64)
            for name in (
                "resident_size",
                "resident_size_peak",
                "device",
                "device_peak",
                "internal",
                "internal_peak",
                "external",
                "external_peak",
                "reusable",
                "reusable_peak",
                "purgeable_volatile_pmap",
                "purgeable_volatile_resident",
                "purgeable_volatile_virtual",
                "compressed",
                "compressed_peak",
                "compressed_lifetime",
                "phys_footprint",
            )
        ],
    ]


def parse_task_vm_info(data: bytes) -> dict | None:
    if len(data) < ctypes.sizeof(TaskVMInfoPrefix):
        return None
    value = TaskVMInfoPrefix.from_buffer_copy(data)
    return {
        "resident_bytes": value.resident_size,
        "peak_resident_bytes": value.resident_size_peak,
        "private_internal_bytes": value.internal,
        "compressed_bytes": value.compressed,
        "physical_footprint_bytes": value.phys_footprint,
    }


def process_memory() -> dict | None:
    if sys.platform != "darwin":
        return None
    try:
        system = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        port = ctypes.c_uint32.in_dll(system, "mach_task_self_").value
        function = system.task_info
        function.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        function.restype = ctypes.c_int32
        buffer = (ctypes.c_uint32 * 256)()
        count = ctypes.c_uint32(len(buffer))
        if function(port, 22, buffer, ctypes.byref(count)) != 0:
            return None
        return parse_task_vm_info(bytes(buffer)[: count.value * 4])
    except (OSError, ValueError, AttributeError):
        return None


def parameter_bytes(module) -> int | None:
    if module is None or not hasattr(module, "parameters"):
        return None
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())
