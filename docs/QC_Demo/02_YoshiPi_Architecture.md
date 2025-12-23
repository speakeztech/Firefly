# YoshiPi Hardware Architecture

> **Platform**: Raspberry Pi Zero 2 W on YoshiPi carrier board
>
> **OS**: Raspberry Pi OS (Debian-based) or minimal Buildroot Linux

---

## Hardware Overview

### Raspberry Pi Zero 2 W

| Specification | Value |
|---------------|-------|
| **Processor** | Broadcom BCM2710A1, quad-core Cortex-A53 (64-bit) @ 1GHz |
| **Memory** | 512MB LPDDR2 SDRAM |
| **Connectivity** | 2.4GHz 802.11 b/g/n Wi-Fi, Bluetooth 4.2 BLE |
| **GPIO** | 40-pin header (directly on YoshiPi carrier) |
| **USB** | Micro USB OTG |

### YoshiPi Carrier Board

| Feature | Specification | Demo Use |
|---------|---------------|----------|
| **Analog Inputs** | 4x 10-bit ADC channels | Avalanche circuit input |
| **GPIO** | 10 general purpose I/O | Status LEDs, control |
| **Relays** | 2x dry contact | Not needed for demo |
| **Display** | Integrated touchscreen | Monitor UI |
| **Proto Area** | Breadboard-style prototyping | Avalanche circuit mount |
| **Expansion** | Mikrobus, Grove I2C, Qwiic I2C | Future sensors |

---

## Avalanche Circuit Integration

### Physical Connection

```
┌─────────────────────────────────────────────────────────────────────┐
│  YoshiPi Carrier Board                                               │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │  ┌─────────────┐                      ┌──────────────────┐   │  │
│  │  │ Proto Area  │                      │ Analog Input 0   │   │  │
│  │  │             │  ──────────────────► │ (ADC Channel 0)  │   │  │
│  │  │ Avalanche   │  Signal wire         │                  │   │  │
│  │  │ Circuit     │                      └──────────────────┘   │  │
│  │  │             │                                              │  │
│  │  │  ┌───┐      │                      ┌──────────────────┐   │  │
│  │  │  │ Z │◄─┐   │                      │ 3.3V Power       │   │  │
│  │  │  └───┘  │   │  ◄─────────────────  │                  │   │  │
│  │  │    │    │   │  Power               └──────────────────┘   │  │
│  │  │    ▼    │   │                                              │  │
│  │  │  Noise  │   │                      ┌──────────────────┐   │  │
│  │  │  Amp    │   │                      │ GND              │   │  │
│  │  │    │    │   │  ◄─────────────────  │                  │   │  │
│  │  │    └────┘   │  Ground              └──────────────────┘   │  │
│  │  └─────────────┘                                              │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Avalanche Circuit (Simplified)

The multi-tiered avalanche circuit generates true random noise from quantum mechanical effects (electron tunneling through reverse-biased PN junctions).

```
     3.3V
      │
      ├──┬──────────────────────────────────────┐
      │  │                                      │
     [R1]│                                     [R4]
      │  │                                      │
      ├──┼───► Stage 1 Noise ───► Amp ───┐     │
      │  │                               │     │
     ┌┴┐ │                               ▼     │
     │Z│ ▼                           ┌───────┐ │
     │1│ Avalanche                   │ Final │ │
     └┬┘ Diode                       │ Stage │─┴──► To ADC
      │                               │ Amp   │
      ▼                               └───────┘
     GND
```

The amplified noise signal (0-3.3V range) connects to Analog Input 0.

---

## Linux Device Access

### ADC via Industrial I/O (IIO) Subsystem

The YoshiPi's ADC appears under the Linux IIO subsystem:

```
/sys/bus/iio/devices/iio:device0/
├── name                        # "yoshipi-adc" or similar
├── in_voltage0_raw             # Channel 0 raw value (0-1023)
├── in_voltage1_raw             # Channel 1 raw value
├── in_voltage2_raw             # Channel 2 raw value
├── in_voltage3_raw             # Channel 3 raw value
├── in_voltage_scale            # Scale factor for mV conversion
├── sampling_frequency          # Current sample rate
└── sampling_frequency_available # Available sample rates
```

**Reading ADC:**
```bash
# Read single sample
cat /sys/bus/iio/devices/iio:device0/in_voltage0_raw
# Returns: 512 (example value, 0-1023 range)

# Read scale
cat /sys/bus/iio/devices/iio:device0/in_voltage_scale
# Returns: 3.222656 (mV per LSB)
```

**High-Speed Sampling:**

For continuous sampling, use the IIO buffer interface:
```
/dev/iio:device0                # Character device for buffered access
```

Or trigger-based sampling via:
```
/sys/bus/iio/devices/iio:device0/buffer/
├── enable                      # 1 to start, 0 to stop
├── length                      # Buffer length in samples
└── watermark                   # Watermark for poll/select
```

### GPIO via gpiochip

The Pi's GPIO appears as:
```
/dev/gpiochip0                  # Main GPIO controller
```

**GPIO ioctl interface:**
```c
// Structures (for reference - Alex generates these)
struct gpioline_info {
    uint32_t line_offset;
    uint32_t flags;
    char name[32];
    char consumer[32];
};

struct gpiohandle_request {
    uint32_t lineoffsets[64];
    uint32_t flags;
    uint8_t default_values[64];
    char consumer_label[32];
    uint32_t lines;
    int fd;
};

// ioctl commands
#define GPIO_GET_LINEINFO_IOCTL    _IOWR(0xB4, 0x02, struct gpioline_info)
#define GPIO_GET_LINEHANDLE_IOCTL  _IOWR(0xB4, 0x03, struct gpiohandle_request)
#define GPIOHANDLE_GET_LINE_VALUES_IOCTL _IOWR(0xB4, 0x08, struct gpiohandle_data)
#define GPIOHANDLE_SET_LINE_VALUES_IOCTL _IOWR(0xB4, 0x09, struct gpiohandle_data)
```

---

## Display Subsystem

### Option 1: WebKitGTK (Preferred)

If GTK and WebKitGTK are available on the Pi Zero 2 W image:

```fsharp
// Same pattern as desktop
let w = Webview.create true
Webview.setTitle w "QC Monitor"
Webview.setSize w 480 320  // Match display resolution
Webview.setHtml w EmbeddedAssets.MonitorHtml
Webview.run w
```

**Considerations:**
- Pi Zero 2 W has limited RAM (512MB)
- WebKitGTK is memory-hungry
- May need minimal X11/Wayland setup
- Consider lightweight alternatives if memory is tight

### Option 2: Framebuffer Direct (Fallback)

If WebKitGTK is too heavy:

```
/dev/fb0                        # Framebuffer device
```

**Framebuffer info:**
```c
struct fb_var_screeninfo {
    uint32_t xres, yres;        // Visible resolution
    uint32_t bits_per_pixel;    // 16, 24, or 32
    // ...
};
```

Can render directly with memory-mapped writes. Would need custom UI rendering (not shared with desktop).

### Option 3: DRM/KMS (Modern)

```
/dev/dri/card0                  # DRM device
```

More complex but better performance. Still not shared with desktop UI.

**Recommendation:** Try WebKitGTK first. If memory is too tight, fall back to a simple framebuffer status display.

---

## USB Gadget Mode

The Pi Zero 2 W supports USB OTG, enabling gadget mode for credential transfer:

```
# Configure USB gadget (CDC ACM - serial port)
modprobe libcomposite
# ... gadget configuration ...
```

Result:
```
/dev/ttyGS0                     # USB gadget serial port (device side)
```

On the desktop, this appears as `/dev/ttyACM0` or similar.

**Credential Transfer Flow:**
```
YoshiPi                         Desktop
   │                               │
   │ /dev/ttyGS0                   │ /dev/ttyACM0
   │     │                         │     │
   └─────┼─────── USB Cable ───────┼─────┘
         │                         │
    write(fd, cred)           read(fd, buf)
```

---

## Memory Map

### Process Memory (Linux User Space)

```
┌─────────────────────────────────────────────────────────────────────┐
│  User Space Process (Firefly-compiled binary)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Stack (grows down)                                           │   │
│  │ • Local variables                                            │   │
│  │ • NativeArray.stackalloc buffers                             │   │
│  │ • Function call frames                                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│                          ▼                                          │
│                       (grows)                                       │
│                                                                     │
│                       (grows)                                       │
│                          ▲                                          │
│                          │                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Heap (optional - stack_only mode avoids this)                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ mmap regions                                                 │   │
│  │ • /dev/fb0 framebuffer (if used)                             │   │
│  │ • Shared memory for IPC (if needed)                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Data segment                                                 │   │
│  │ • Embedded HTML/JS/CSS (WebView assets)                      │   │
│  │ • Static string constants                                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Text segment (read-only, executable)                         │   │
│  │ • Compiled F# code                                           │   │
│  │ • PQC algorithm implementations                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Power Considerations

### Pi Zero 2 W Power Budget

| Component | Typical Draw |
|-----------|-------------|
| Pi Zero 2 W (idle) | ~100mA @ 5V |
| Pi Zero 2 W (load) | ~300mA @ 5V |
| Display | ~50-100mA |
| Avalanche circuit | ~10mA |
| **Total** | ~400-500mA @ 5V |

**Power Source Options:**
- USB power (standard micro USB)
- Battery pack (portable demo)
- Bench supply (lab demo)

---

## Software Stack

### Minimal Linux Image

For demo, consider a minimal image:

```
Base: Raspberry Pi OS Lite (no desktop)
Add:
  - WebKitGTK (if used for UI)
  - GTK3 dependencies
  - libiio (for ADC access)
  - libgpiod (for GPIO)

Remove:
  - Desktop environment
  - Unnecessary services
  - Development tools
```

Target image size: < 1GB

### Required Kernel Modules

```
# ADC
iio-core
yoshipi-adc  # Or whatever the carrier's ADC driver is

# GPIO
gpio-core

# USB Gadget
libcomposite
usb_f_acm

# Display (if using framebuffer)
fb
drm
vc4
```

---

## Cross-References

- [01_YoshiPi_Demo_Strategy](./01_YoshiPi_Demo_Strategy.md) - Overall strategy
- [03_Linux_Hardware_Bindings](./03_Linux_Hardware_Bindings.md) - Alloy Platform.Bindings for Linux hardware
- [04_PostQuantum_Architecture](./04_PostQuantum_Architecture.md) - PQC algorithms
