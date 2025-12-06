/// STM32L5 - Hardware abstraction layer for STM32L5 series microcontrollers
/// Provides type-safe register access with zero runtime cost
module STM32L5

open Alloy.Memory

// =============================================================================
// Memory-Mapped Register Addresses
// =============================================================================

module Addresses =
    // Peripheral base addresses
    let PERIPH_BASE       = 0x40000000u
    let APB1PERIPH_BASE   = PERIPH_BASE
    let APB2PERIPH_BASE   = PERIPH_BASE + 0x00010000u
    let AHB1PERIPH_BASE   = PERIPH_BASE + 0x00020000u
    let AHB2PERIPH_BASE   = PERIPH_BASE + 0x02020000u

    // RCC (Reset and Clock Control)
    let RCC_BASE          = AHB1PERIPH_BASE + 0x00001000u
    let RCC_AHB2ENR       = RCC_BASE + 0x4Cu  // AHB2 peripheral clock enable

    // GPIO port base addresses
    let GPIOA_BASE        = AHB2PERIPH_BASE + 0x00000000u
    let GPIOB_BASE        = AHB2PERIPH_BASE + 0x00000400u
    let GPIOC_BASE        = AHB2PERIPH_BASE + 0x00000800u
    let GPIOD_BASE        = AHB2PERIPH_BASE + 0x00000C00u
    let GPIOE_BASE        = AHB2PERIPH_BASE + 0x00001000u
    let GPIOF_BASE        = AHB2PERIPH_BASE + 0x00001400u
    let GPIOG_BASE        = AHB2PERIPH_BASE + 0x00001800u
    let GPIOH_BASE        = AHB2PERIPH_BASE + 0x00001C00u

// =============================================================================
// GPIO Register Layout
// =============================================================================

/// GPIO Register offsets (from port base address)
module GPIO =
    /// Register offsets
    let MODER_OFFSET   = 0x00u  // Mode register
    let OTYPER_OFFSET  = 0x04u  // Output type register
    let OSPEEDR_OFFSET = 0x08u  // Output speed register
    let PUPDR_OFFSET   = 0x0Cu  // Pull-up/pull-down register
    let IDR_OFFSET     = 0x10u  // Input data register
    let ODR_OFFSET     = 0x14u  // Output data register
    let BSRR_OFFSET    = 0x18u  // Bit set/reset register
    let LCKR_OFFSET    = 0x1Cu  // Lock register
    let AFRL_OFFSET    = 0x20u  // Alternate function low register
    let AFRH_OFFSET    = 0x24u  // Alternate function high register

    /// Pin mode values
    type PinMode =
        | Input     = 0b00u
        | Output    = 0b01u
        | Alternate = 0b10u
        | Analog    = 0b11u

    /// Output type values
    type OutputType =
        | PushPull  = 0u
        | OpenDrain = 1u

    /// Output speed values
    type OutputSpeed =
        | Low      = 0b00u
        | Medium   = 0b01u
        | High     = 0b10u
        | VeryHigh = 0b11u

    /// Pull-up/pull-down values
    type PullMode =
        | None     = 0b00u
        | PullUp   = 0b01u
        | PullDown = 0b10u

    /// Get base address for a GPIO port
    let inline portBase (port: char) : uint32 =
        match port with
        | 'A' -> Addresses.GPIOA_BASE
        | 'B' -> Addresses.GPIOB_BASE
        | 'C' -> Addresses.GPIOC_BASE
        | 'D' -> Addresses.GPIOD_BASE
        | 'E' -> Addresses.GPIOE_BASE
        | 'F' -> Addresses.GPIOF_BASE
        | 'G' -> Addresses.GPIOG_BASE
        | 'H' -> Addresses.GPIOH_BASE
        | _ -> Addresses.GPIOA_BASE  // Default to GPIOA

    /// Enable clock for a GPIO port
    let inline enableClock (port: char) =
        let bit = int port - int 'A'
        let addr = nativeint Addresses.RCC_AHB2ENR
        let current = Ptr.read<uint32> addr
        Ptr.write addr (current ||| (1u <<< bit))

    /// Configure pin mode (input, output, alternate, analog)
    let inline configureMode (port: char) (pin: int) (mode: PinMode) =
        let baseAddr = portBase port
        let addr = nativeint (baseAddr + MODER_OFFSET)
        let shift = pin * 2
        let mask = ~~~(3u <<< shift)
        let current = Ptr.read<uint32> addr
        Ptr.write addr ((current &&& mask) ||| ((uint32 mode) <<< shift))

    /// Configure output type
    let inline configureOutputType (port: char) (pin: int) (otype: OutputType) =
        let baseAddr = portBase port
        let addr = nativeint (baseAddr + OTYPER_OFFSET)
        let current = Ptr.read<uint32> addr
        if otype = OutputType.OpenDrain then
            Ptr.write addr (current ||| (1u <<< pin))
        else
            Ptr.write addr (current &&& ~~~(1u <<< pin))

    /// Configure output speed
    let inline configureSpeed (port: char) (pin: int) (speed: OutputSpeed) =
        let baseAddr = portBase port
        let addr = nativeint (baseAddr + OSPEEDR_OFFSET)
        let shift = pin * 2
        let mask = ~~~(3u <<< shift)
        let current = Ptr.read<uint32> addr
        Ptr.write addr ((current &&& mask) ||| ((uint32 speed) <<< shift))

    /// Configure pull-up/pull-down
    let inline configurePull (port: char) (pin: int) (pull: PullMode) =
        let baseAddr = portBase port
        let addr = nativeint (baseAddr + PUPDR_OFFSET)
        let shift = pin * 2
        let mask = ~~~(3u <<< shift)
        let current = Ptr.read<uint32> addr
        Ptr.write addr ((current &&& mask) ||| ((uint32 pull) <<< shift))

    /// Set pin high
    let inline setPin (port: char) (pin: int) =
        let baseAddr = portBase port
        let addr = nativeint (baseAddr + BSRR_OFFSET)
        Ptr.write<uint32> addr (1u <<< pin)

    /// Set pin low
    let inline clearPin (port: char) (pin: int) =
        let baseAddr = portBase port
        let addr = nativeint (baseAddr + BSRR_OFFSET)
        Ptr.write<uint32> addr (1u <<< (pin + 16))

    /// Toggle pin
    let inline togglePin (port: char) (pin: int) =
        let baseAddr = portBase port
        let odrAddr = nativeint (baseAddr + ODR_OFFSET)
        let current = Ptr.read<uint32> odrAddr
        if (current &&& (1u <<< pin)) <> 0u then
            clearPin port pin
        else
            setPin port pin

    /// Read pin state
    let inline readPin (port: char) (pin: int) : bool =
        let baseAddr = portBase port
        let addr = nativeint (baseAddr + IDR_OFFSET)
        let value = Ptr.read<uint32> addr
        (value &&& (1u <<< pin)) <> 0u

    /// Configure pin as output with common settings
    let inline configureAsOutput (port: char) (pin: int) =
        enableClock port
        configureMode port pin PinMode.Output
        configureOutputType port pin OutputType.PushPull
        configureSpeed port pin OutputSpeed.Low
        configurePull port pin PullMode.None

    /// Configure pin as input with optional pull
    let inline configureAsInput (port: char) (pin: int) (pull: PullMode) =
        enableClock port
        configureMode port pin PinMode.Input
        configurePull port pin pull

// =============================================================================
// Simple delay function (busy wait)
// =============================================================================

module Delay =
    /// Simple busy-wait delay
    /// Note: This is not accurate timing - just for simple demos
    /// For accurate timing, use hardware timers
    let inline cycles (count: int) =
        let mutable i = 0
        while i < count do
            i <- i + 1

    /// Approximate millisecond delay (at ~80MHz)
    /// This is very rough - real code should use SysTick or timers
    let inline ms (milliseconds: int) =
        cycles (milliseconds * 10000)
