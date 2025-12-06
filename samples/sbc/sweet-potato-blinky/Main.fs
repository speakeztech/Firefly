/// Libre Sweet Potato Blinky - ARM64 SBC blink demo
/// Demonstrates GPIO control on Allwinner H6-based board
///
/// Hardware:
/// - Board: Libre Sweet Potato (or compatible H6 board)
/// - This is a bare-metal example that runs without Linux
/// - Assumes U-Boot or similar has initialized DDR and basic peripherals
///
/// Note: Pin assignments depend on your specific board layout.
/// Check your board schematic for LED locations.
module SweetPotatoBlinky

open AllwinnerH6
open AllwinnerH6.GPIO
open AllwinnerH6.Delay

// =============================================================================
// Board-specific configuration
// =============================================================================

/// GPIO pins for LEDs (adjust based on actual board)
/// These are example pins - verify with your board schematic
module LED =
    // Example: Status LED on PL10 (R_PIO domain)
    // Note: For simplicity, using main PIO domain GPIO
    // Real implementation would handle R_PIO separately

    // Green LED - example on PC7
    let GREEN_PORT = 'C'
    let GREEN_PIN = 7

    // Red LED - example on PC8
    let RED_PORT = 'C'
    let RED_PIN = 8

    // Blue LED - example on PC9
    let BLUE_PORT = 'C'
    let BLUE_PIN = 9

    /// Initialize all LEDs as outputs
    let initAll() =
        configureAsOutput GREEN_PORT GREEN_PIN
        configureAsOutput RED_PORT RED_PIN
        configureAsOutput BLUE_PORT BLUE_PIN

    /// Green LED control
    let greenOn() = setPin GREEN_PORT GREEN_PIN
    let greenOff() = clearPin GREEN_PORT GREEN_PIN
    let greenToggle() = togglePin GREEN_PORT GREEN_PIN

    /// Red LED control
    let redOn() = setPin RED_PORT RED_PIN
    let redOff() = clearPin RED_PORT RED_PIN
    let redToggle() = togglePin RED_PORT RED_PIN

    /// Blue LED control
    let blueOn() = setPin BLUE_PORT BLUE_PIN
    let blueOff() = clearPin BLUE_PORT BLUE_PIN
    let blueToggle() = togglePin BLUE_PORT BLUE_PIN

    /// All LEDs off
    let allOff() =
        greenOff()
        redOff()
        blueOff()

    /// All LEDs on
    let allOn() =
        greenOn()
        redOn()
        blueOn()

// =============================================================================
// Blink patterns
// =============================================================================

/// Simple blink - toggle LED at fixed interval
let simpleBlink() =
    LED.initAll()
    LED.allOff()

    while true do
        LED.greenToggle()
        ms 500

/// RGB chase pattern
let rgbChase() =
    LED.initAll()
    LED.allOff()

    while true do
        // Green
        LED.greenOn()
        LED.redOff()
        LED.blueOff()
        ms 300

        // Red
        LED.greenOff()
        LED.redOn()
        LED.blueOff()
        ms 300

        // Blue
        LED.greenOff()
        LED.redOff()
        LED.blueOn()
        ms 300

/// All LEDs blinking together
let allBlink() =
    LED.initAll()

    while true do
        LED.allOn()
        ms 250
        LED.allOff()
        ms 250

/// Binary counter pattern (3 bits = 0-7)
let binaryCounter() =
    LED.initAll()
    LED.allOff()

    let mutable count = 0

    while true do
        // Display count on LEDs
        if (count &&& 1) <> 0 then LED.greenOn() else LED.greenOff()
        if (count &&& 2) <> 0 then LED.redOn() else LED.redOff()
        if (count &&& 4) <> 0 then LED.blueOn() else LED.blueOff()

        ms 500

        count <- (count + 1) &&& 7  // Wrap at 7

// =============================================================================
// Entry point
// =============================================================================

/// Main entry point
/// Called after startup code initializes runtime
[<EntryPoint>]
let main argv =
    // Run simple blink pattern
    // Change to rgbChase(), allBlink(), or binaryCounter() for other patterns
    simpleBlink()

    // Should never reach here
    0
