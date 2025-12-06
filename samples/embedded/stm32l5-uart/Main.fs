/// STM32L5 UART Echo - Serial communication demo
/// Demonstrates UART communication on NUCLEO-L552ZE-Q
///
/// Hardware:
/// - USART2 is connected to ST-Link VCP (Virtual COM Port)
/// - Connect via USB and use a serial terminal (115200 8N1)
module UARTEcho

open STM32L5
open STM32L5.GPIO
open STM32L5.UART
open STM32L5.Delay

// =============================================================================
// Application
// =============================================================================

/// Send a welcome message
let sendWelcome() =
    USART2.sendLine ""
    USART2.sendLine "=================================="
    USART2.sendLine "  Firefly UART Echo Demo"
    USART2.sendLine "  NUCLEO-L552ZE-Q"
    USART2.sendLine "=================================="
    USART2.sendLine ""
    USART2.sendLine "Type something and press Enter..."
    USART2.sendLine ""

/// Echo received characters back with LED indication
let echoLoop() =
    // Initialize LED for activity indication
    GPIO.configureAsOutput LED.LD1_PORT LED.LD1_PIN

    use buffer = Alloy.Memory.stackBuffer<byte> 256
    let mutable bufferPos = 0

    while true do
        match USART2.tryReceive() with
        | Some byte ->
            // Toggle LED on activity
            GPIO.togglePin LED.LD1_PORT LED.LD1_PIN

            // Echo the character
            USART2.send (string (char byte))

            // Handle line endings
            if byte = byte '\r' || byte = byte '\n' then
                USART2.sendLine ""
                USART2.send "> "
                bufferPos <- 0
            else
                // Store in buffer (simple line buffer)
                if bufferPos < 255 then
                    buffer.[bufferPos] <- byte
                    bufferPos <- bufferPos + 1
        | None ->
            // No data, small delay to reduce CPU usage
            ()

/// Main entry point
[<EntryPoint>]
let main argv =
    // Initialize USART2 at 115200 baud
    USART2.initDefault()

    // Send welcome message
    sendWelcome()

    // Run echo loop forever
    echoLoop()

    0
