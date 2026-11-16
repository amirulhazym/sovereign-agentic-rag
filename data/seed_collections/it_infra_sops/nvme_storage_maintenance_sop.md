# Enterprise NVMe & Server Storage Maintenance SOP

## 1.1 Approved NVMe Hardware Specification Matrix

| Drive Model | Form Factor | Capacity | Endurance (DWPD) | Max Case Temp | Minimum Firmware |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Enterprise-NVMe-U2-3840 | 2.5-inch U.2 | 3.84 TB | 3.0 DWPD | 70C | FW-4.8.2-PROD |
| Enterprise-NVMe-U2-7680 | 2.5-inch U.2 | 7.68 TB | 1.0 DWPD | 70C | FW-4.8.2-PROD |
| Boot-M2-Enterprise-480 | M.2 2280 | 480 GB | 3.0 DWPD | 75C | FW-2.1.9-BOOT |

All production storage nodes run PCIe Gen4 x4 backplanes with dual redundant PSUs and hot-swappable cooling fan modules.

<!-- page: 2 -->
## 2.1 NVMe Hot-Swap Pre-Conditions & Safety Checks

Before pulling any failed U.2 NVMe bay from a live storage chassis, engineers MUST verify all three pre-conditions:
1. Verify the physical bay fault LED indicator is solid amber (never pull a bay showing blinking green I/O activity).
2. Run `nvme list` and `lsblk -o NAME,SERIAL,STATE` via SSH to confirm namespace detachment and match the serial number printed on the tray latch.
3. Confirm chassis ambient inlet temperature remains at or below 35C during single-fan maintenance windows.

## 2.2 Storage Error Code Reference

- **`ERR_NVME_0x04`:** Indicates controller PCIe link degradation or lane downshift (`Gen4 -> Gen2`). Action: Reseat the U.2 drive carrier in the backplane slot and verify link speed via `lspci -vv` prior to initiating an RMA.
- **`ERR_NVME_0x12`:** Indicates SMART Available Spare fell below 10% (`media_wearout_threshold`). Action: Schedule immediate RAID-10 rebuild replacement within 24 hours.
- **`ERR_FAN_0x09`:** Dual cooling fan module RPM drop below 3,200 RPM. Action: Replace hot-swap fan cartridge within 15 minutes to prevent thermal throttling.

<!-- page: 3 -->
## 3.1 RAID-10 Rebuild & Emergency Thermal Shutdown SOP

1. After inserting a replacement `Enterprise-NVMe-U2-3840` drive with firmware `FW-4.8.2-PROD`, verify rebuild progress using `mdadm --detail /dev/md0`.
2. During active array rebuild, throttle background scrub I/O to 150 MB/s (`echo 150000 > /proc/sys/dev/raid/speed_limit_max`) so customer-facing p99 read latency stays below 12 ms.
3. If NVMe SMART composite temperature reaches 75C (`Critical Composite Temp`), execute graceful service drain (`systemctl stop rag-worker`) and power down the chassis cleanly to prevent silent data corruption.
