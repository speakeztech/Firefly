/// STM32L5.UART - UART/USART driver for STM32L5 series
/// Provides type-safe serial communication
module STM32L5.UART

open Alloy.Memory
open STM32L5

// =============================================================================
// UART Register Addresses
// =============================================================================

module Addresses =
    // USART/UART base addresses
    let USART1_BASE = 0x40013800u
    let USART2_BASE = 0x40004400u
    let USART3_BASE = 0x40004800u
    let UART4_BASE  = 0x40004C00u
    let UART5_BASE  = 0x40005000u
    let LPUART1_BASE = 0x40008000u

    // RCC registers for enabling UART clocks
    let RCC_APB1ENR1 = 0x40021058u
    let RCC_APB1ENR2 = 0x4002105Cu
    let RCC_APB2ENR  = 0x40021060u

// =============================================================================
// UART Register Layout
// =============================================================================

/// UART Register offsets
module Registers =
    let CR1   = 0x00u  // Control register 1
    let CR2   = 0x04u  // Control register 2
    let CR3   = 0x08u  // Control register 3
    let BRR   = 0x0Cu  // Baud rate register
    let GTPR  = 0x10u  // Guard time and prescaler
    let RTOR  = 0x14u  // Receiver timeout
    let RQR   = 0x18u  // Request register
    let ISR   = 0x1Cu  // Interrupt and status register
    let ICR   = 0x20u  // Interrupt clear register
    let RDR   = 0x24u  // Receive data register
    let TDR   = 0x28u  // Transmit data register
    let PRESC = 0x2Cu  // Prescaler register

/// CR1 bit definitions
module CR1 =
    let UE     = 0       // UART enable
    let UESM   = 1       // UART enable in stop mode
    let RE     = 2       // Receiver enable
    let TE     = 3       // Transmitter enable
    let IDLEIE = 4       // IDLE interrupt enable
    let RXNEIE = 5       // RXNE interrupt enable
    let TCIE   = 6       // Transmission complete interrupt enable
    let TXEIE  = 7       // TXE interrupt enable
    let PEIE   = 8       // PE interrupt enable
    let PS     = 9       // Parity selection
    let PCE    = 10      // Parity control enable
    let WAKE   = 11      // Receiver wakeup method
    let M0     = 12      // Word length bit 0
    let MME    = 13      // Mute mode enable
    let CMIE   = 14      // Character match interrupt enable
    let OVER8  = 15      // Oversampling mode
    let M1     = 28      // Word length bit 1

/// ISR bit definitions
module ISR =
    let PE    = 0        // Parity error
    let FE    = 1        // Framing error
    let NE    = 2        // Noise error
    let ORE   = 3        // Overrun error
    let IDLE  = 4        // Idle line detected
    let RXNE  = 5        // Read data register not empty
    let TC    = 6        // Transmission complete
    let TXE   = 7        // Transmit data register empty
    let LBDF  = 8        // LIN break detection flag
    let CTSIF = 9        // CTS interrupt flag
    let CTS   = 10       // CTS flag
    let RTOF  = 11       // Receiver timeout
    let EOBF  = 12       // End of block flag
    let ABRE  = 14       // Auto baud rate error
    let ABRF  = 15       // Auto baud rate flag
    let BUSY  = 16       // Busy flag
    let CMF   = 17       // Character match flag
    let SBKF  = 18       // Send break flag
    let TEACK = 21       // Transmit enable acknowledge
    let REACK = 22       // Receive enable acknowledge

// =============================================================================
// UART Configuration
// =============================================================================

/// Parity options
type Parity =
    | None
    | Even
    | Odd

/// Stop bits
type StopBits =
    | One
    | Two

/// Word length
type WordLength =
    | Bits7
    | Bits8
    | Bits9

/// UART configuration
type UartConfig = {
    BaudRate: int
    WordLength: WordLength
    Parity: Parity
    StopBits: StopBits
}

/// Default configuration (115200 8N1)
let defaultConfig = {
    BaudRate = 115200
    WordLength = Bits8
    Parity = None
    StopBits = One
}

// =============================================================================
// UART Operations
// =============================================================================

/// UART port handle
type UartPort = {
    BaseAddress: uint32
    Config: UartConfig
}

/// Enable clock for UART
let private enableClock (baseAddr: uint32) =
    match baseAddr with
    | addr when addr = Addresses.USART1_BASE ->
        let rcc = nativeint Addresses.RCC_APB2ENR
        let current = Ptr.read<uint32> rcc
        Ptr.write rcc (current ||| (1u <<< 14))
    | addr when addr = Addresses.USART2_BASE ->
        let rcc = nativeint Addresses.RCC_APB1ENR1
        let current = Ptr.read<uint32> rcc
        Ptr.write rcc (current ||| (1u <<< 17))
    | addr when addr = Addresses.USART3_BASE ->
        let rcc = nativeint Addresses.RCC_APB1ENR1
        let current = Ptr.read<uint32> rcc
        Ptr.write rcc (current ||| (1u <<< 18))
    | addr when addr = Addresses.LPUART1_BASE ->
        let rcc = nativeint Addresses.RCC_APB1ENR2
        let current = Ptr.read<uint32> rcc
        Ptr.write rcc (current ||| (1u <<< 0))
    | _ -> ()

/// Calculate BRR value for given baud rate
/// Assumes 80MHz clock (default after reset)
let private calculateBRR (baudRate: int) (over8: bool) : uint32 =
    let clockFreq = 80000000u  // 80 MHz
    if over8 then
        // OVER8 = 1: BRR = 2 * USARTDIV
        (2u * clockFreq + uint32 baudRate / 2u) / uint32 baudRate
    else
        // OVER8 = 0: BRR = USARTDIV
        (clockFreq + uint32 baudRate / 2u) / uint32 baudRate

/// Initialize UART port
let init (baseAddr: uint32) (config: UartConfig) : UartPort =
    // Enable clock
    enableClock baseAddr

    let base = nativeint baseAddr

    // Disable UART during configuration
    let cr1Addr = base + nativeint Registers.CR1
    Ptr.write<uint32> cr1Addr 0u

    // Configure baud rate
    let brrAddr = base + nativeint Registers.BRR
    Ptr.write brrAddr (calculateBRR config.BaudRate false)

    // Configure CR1 (word length, parity, etc.)
    let mutable cr1 = 0u

    // Word length
    match config.WordLength with
    | Bits7 -> cr1 <- cr1 ||| (1u <<< CR1.M1)
    | Bits8 -> ()  // M[1:0] = 00
    | Bits9 -> cr1 <- cr1 ||| (1u <<< CR1.M0)

    // Parity
    match config.Parity with
    | None -> ()
    | Even -> cr1 <- cr1 ||| (1u <<< CR1.PCE)
    | Odd -> cr1 <- cr1 ||| (1u <<< CR1.PCE) ||| (1u <<< CR1.PS)

    // Enable transmitter and receiver
    cr1 <- cr1 ||| (1u <<< CR1.TE) ||| (1u <<< CR1.RE)

    // Enable UART
    cr1 <- cr1 ||| (1u <<< CR1.UE)

    Ptr.write cr1Addr cr1

    // Configure CR2 (stop bits)
    let cr2Addr = base + nativeint Registers.CR2
    let cr2 =
        match config.StopBits with
        | One -> 0u
        | Two -> 2u <<< 12
    Ptr.write cr2Addr cr2

    { BaseAddress = baseAddr; Config = config }

/// Check if transmit buffer is empty
let inline canTransmit (port: UartPort) : bool =
    let isrAddr = nativeint (port.BaseAddress + Registers.ISR)
    let isr = Ptr.read<uint32> isrAddr
    (isr &&& (1u <<< ISR.TXE)) <> 0u

/// Check if receive buffer has data
let inline hasData (port: UartPort) : bool =
    let isrAddr = nativeint (port.BaseAddress + Registers.ISR)
    let isr = Ptr.read<uint32> isrAddr
    (isr &&& (1u <<< ISR.RXNE)) <> 0u

/// Transmit a single byte (blocking)
let transmitByte (port: UartPort) (data: byte) =
    // Wait for TXE
    while not (canTransmit port) do ()

    // Write data
    let tdrAddr = nativeint (port.BaseAddress + Registers.TDR)
    Ptr.write<uint32> tdrAddr (uint32 data)

/// Receive a single byte (blocking)
let receiveByte (port: UartPort) : byte =
    // Wait for RXNE
    while not (hasData port) do ()

    // Read data
    let rdrAddr = nativeint (port.BaseAddress + Registers.RDR)
    byte (Ptr.read<uint32> rdrAddr)

/// Transmit a string
let transmitString (port: UartPort) (s: string) =
    for c in s do
        transmitByte port (byte c)

/// Transmit with newline
let transmitLine (port: UartPort) (s: string) =
    transmitString port s
    transmitByte port (byte '\r')
    transmitByte port (byte '\n')

/// Try to receive a byte (non-blocking)
let tryReceiveByte (port: UartPort) : byte option =
    if hasData port then
        Some (receiveByte port)
    else
        None

// =============================================================================
// Convenience instances
// =============================================================================

/// USART2 - Connected to ST-Link VCP on NUCLEO boards
module USART2 =
    let mutable private port = Unchecked.defaultof<UartPort>

    /// Initialize USART2 with default settings
    let initDefault() =
        port <- init Addresses.USART2_BASE defaultConfig

    /// Initialize USART2 with custom baud rate
    let initWithBaud (baudRate: int) =
        port <- init Addresses.USART2_BASE { defaultConfig with BaudRate = baudRate }

    /// Send a string
    let send (s: string) = transmitString port s

    /// Send a line
    let sendLine (s: string) = transmitLine port s

    /// Receive a byte (blocking)
    let receive() = receiveByte port

    /// Try to receive (non-blocking)
    let tryReceive() = tryReceiveByte port
