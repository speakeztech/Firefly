/// AllwinnerH6 - Hardware abstraction layer for Allwinner H6 SoC
/// Used in Libre Sweet Potato and similar ARM64 boards
module AllwinnerH6

open Alloy.Memory

// =============================================================================
// Memory-Mapped Register Addresses
// =============================================================================

module Addresses =
    // Peripheral base addresses for H6
    let DE_BASE           = 0x01000000UL  // Display Engine
    let GPU_BASE          = 0x01800000UL  // GPU
    let CE_BASE           = 0x01904000UL  // Crypto Engine
    let EMCE_BASE         = 0x01905000UL  // EMCE

    // System control
    let SYSCON_BASE       = 0x03000000UL  // System Control
    let CCU_BASE          = 0x03001000UL  // Clock Control Unit
    let DMA_BASE          = 0x03002000UL  // DMA Controller
    let MSGBOX_BASE       = 0x03003000UL  // Message Box
    let SPINLOCK_BASE     = 0x03004000UL  // Spinlock
    let HSTIMER_BASE      = 0x03005000UL  // High-Speed Timer

    // GPIO (PIO)
    let PIO_BASE          = 0x0300B000UL  // Port I/O Controller
    let R_PIO_BASE        = 0x07022000UL  // R_PIO (always-on domain)

    // UART
    let UART0_BASE        = 0x05000000UL
    let UART1_BASE        = 0x05000400UL
    let UART2_BASE        = 0x05000800UL
    let UART3_BASE        = 0x05000C00UL

    // I2C
    let TWI0_BASE         = 0x05002000UL
    let TWI1_BASE         = 0x05002400UL
    let TWI2_BASE         = 0x05002800UL

    // SPI
    let SPI0_BASE         = 0x05010000UL
    let SPI1_BASE         = 0x05011000UL

// =============================================================================
// Clock Control Unit (CCU)
// =============================================================================

module CCU =
    open Addresses

    /// Register offsets
    let PLL_CPUX_CTRL     = 0x00u
    let PLL_DDR0_CTRL     = 0x10u
    let PLL_PERIPH0_CTRL  = 0x20u
    let PLL_GPU_CTRL      = 0x30u
    let PLL_VIDEO0_CTRL   = 0x40u

    let BUS_CLK_GATING0   = 0x60u
    let BUS_CLK_GATING1   = 0x64u
    let BUS_CLK_GATING2   = 0x68u
    let BUS_CLK_GATING3   = 0x6Cu

    let BUS_SOFT_RST0     = 0x70u
    let BUS_SOFT_RST1     = 0x74u
    let BUS_SOFT_RST2     = 0x78u
    let BUS_SOFT_RST3     = 0x7Cu

    /// Enable clock for a peripheral
    let inline enableClock (register: uint32) (bit: int) =
        let addr = nativeint (CCU_BASE + uint64 register)
        let current = Ptr.read<uint32> addr
        Ptr.write addr (current ||| (1u <<< bit))

    /// Disable clock for a peripheral
    let inline disableClock (register: uint32) (bit: int) =
        let addr = nativeint (CCU_BASE + uint64 register)
        let current = Ptr.read<uint32> addr
        Ptr.write addr (current &&& ~~~(1u <<< bit))

// =============================================================================
// GPIO (PIO) Controller
// =============================================================================

module GPIO =
    open Addresses

    /// Port configuration register offsets
    /// Each port has: CFG0-3 (mode), DAT (data), DRV0-1 (drive), PUL0-1 (pull)
    let inline portOffset (port: char) : uint64 =
        uint64 ((int port - int 'A') * 0x24)

    /// Register offsets within each port
    let CFG0_OFFSET = 0x00u   // Pins 0-7 config (4 bits each)
    let CFG1_OFFSET = 0x04u   // Pins 8-15 config
    let CFG2_OFFSET = 0x08u   // Pins 16-23 config
    let CFG3_OFFSET = 0x0Cu   // Pins 24-31 config
    let DAT_OFFSET  = 0x10u   // Data register
    let DRV0_OFFSET = 0x14u   // Drive strength 0-15
    let DRV1_OFFSET = 0x18u   // Drive strength 16-31
    let PUL0_OFFSET = 0x1Cu   // Pull 0-15
    let PUL1_OFFSET = 0x20u   // Pull 16-31

    /// Pin function/mode values (H6-specific)
    type PinFunction =
        | Input    = 0b000u
        | Output   = 0b001u
        | Func2    = 0b010u  // Alternate function 2
        | Func3    = 0b011u  // Alternate function 3
        | Func4    = 0b100u  // Alternate function 4
        | Func5    = 0b101u  // Alternate function 5
        | Func6    = 0b110u  // Alternate function 6
        | Disabled = 0b111u

    /// Pull-up/pull-down values
    type PullMode =
        | None     = 0b00u
        | PullUp   = 0b01u
        | PullDown = 0b10u

    /// Drive strength values
    type DriveStrength =
        | Level0 = 0b00u  // Weakest
        | Level1 = 0b01u
        | Level2 = 0b10u
        | Level3 = 0b11u  // Strongest

    /// Configure pin function
    let inline configureFunction (port: char) (pin: int) (func: PinFunction) =
        let baseAddr = PIO_BASE + portOffset port
        let cfgReg = (pin / 8) * 4  // CFG0, CFG1, CFG2, or CFG3
        let addr = nativeint (baseAddr + uint64 cfgReg)
        let shift = (pin % 8) * 4
        let mask = ~~~(0xFu <<< shift)
        let current = Ptr.read<uint32> addr
        Ptr.write addr ((current &&& mask) ||| ((uint32 func) <<< shift))

    /// Configure pull mode
    let inline configurePull (port: char) (pin: int) (pull: PullMode) =
        let baseAddr = PIO_BASE + portOffset port
        let pulReg = if pin < 16 then PUL0_OFFSET else PUL1_OFFSET
        let addr = nativeint (baseAddr + uint64 pulReg)
        let shift = (pin % 16) * 2
        let mask = ~~~(3u <<< shift)
        let current = Ptr.read<uint32> addr
        Ptr.write addr ((current &&& mask) ||| ((uint32 pull) <<< shift))

    /// Configure drive strength
    let inline configureDrive (port: char) (pin: int) (drive: DriveStrength) =
        let baseAddr = PIO_BASE + portOffset port
        let drvReg = if pin < 16 then DRV0_OFFSET else DRV1_OFFSET
        let addr = nativeint (baseAddr + uint64 drvReg)
        let shift = (pin % 16) * 2
        let mask = ~~~(3u <<< shift)
        let current = Ptr.read<uint32> addr
        Ptr.write addr ((current &&& mask) ||| ((uint32 drive) <<< shift))

    /// Set pin high
    let inline setPin (port: char) (pin: int) =
        let baseAddr = PIO_BASE + portOffset port
        let addr = nativeint (baseAddr + uint64 DAT_OFFSET)
        let current = Ptr.read<uint32> addr
        Ptr.write addr (current ||| (1u <<< pin))

    /// Set pin low
    let inline clearPin (port: char) (pin: int) =
        let baseAddr = PIO_BASE + portOffset port
        let addr = nativeint (baseAddr + uint64 DAT_OFFSET)
        let current = Ptr.read<uint32> addr
        Ptr.write addr (current &&& ~~~(1u <<< pin))

    /// Toggle pin
    let inline togglePin (port: char) (pin: int) =
        let baseAddr = PIO_BASE + portOffset port
        let addr = nativeint (baseAddr + uint64 DAT_OFFSET)
        let current = Ptr.read<uint32> addr
        Ptr.write addr (current ^^^ (1u <<< pin))

    /// Read pin state
    let inline readPin (port: char) (pin: int) : bool =
        let baseAddr = PIO_BASE + portOffset port
        let addr = nativeint (baseAddr + uint64 DAT_OFFSET)
        let value = Ptr.read<uint32> addr
        (value &&& (1u <<< pin)) <> 0u

    /// Configure pin as output
    let inline configureAsOutput (port: char) (pin: int) =
        configureFunction port pin PinFunction.Output
        configureDrive port pin DriveStrength.Level2
        configurePull port pin PullMode.None

    /// Configure pin as input with pull mode
    let inline configureAsInput (port: char) (pin: int) (pull: PullMode) =
        configureFunction port pin PinFunction.Input
        configurePull port pin pull

// =============================================================================
// Simple delay function
// =============================================================================

module Delay =
    /// Simple busy-wait delay
    let inline cycles (count: int64) =
        let mutable i = 0L
        while i < count do
            i <- i + 1L

    /// Approximate millisecond delay (at ~1.8GHz)
    /// Very rough - real code should use timer hardware
    let inline ms (milliseconds: int) =
        cycles (int64 milliseconds * 500000L)
