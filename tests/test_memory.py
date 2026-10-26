import ctypes

from category_matcher.memory import TaskVMInfoPrefix, parameter_bytes, parse_task_vm_info


def test_vm_info_parser_uses_sdk_layout_and_rejects_truncated_records():
    assert ctypes.sizeof(TaskVMInfoPrefix) == 152
    assert TaskVMInfoPrefix.compressed.offset == 120
    assert TaskVMInfoPrefix.phys_footprint.offset == 144
    source = TaskVMInfoPrefix()
    source.resident_size = 100
    source.resident_size_peak = 140
    source.internal = 70
    source.compressed = 20
    source.phys_footprint = 110
    assert parse_task_vm_info(bytes(source)) == {
        "resident_bytes": 100,
        "peak_resident_bytes": 140,
        "private_internal_bytes": 70,
        "compressed_bytes": 20,
        "physical_footprint_bytes": 110,
    }
    assert parse_task_vm_info(bytes(source)[:20]) is None
    assert parameter_bytes(None) is None
