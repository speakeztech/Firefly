/// STM32L5 Blinky - First embedded Firefly sample
/// Demonstrates GPIO control on NUCLEO-L552ZE-Q board
///
/// Hardware:
/// - Board: NUCLEO-L552ZE-Q
/// - LED: User LED on PC7 (directly from schematic)
/// - Note: The board also has LEDs on PA5 (LD2), PB7 (LD3)
module Blinky

open STM32L5
open STM32L5.GPIO
open STM32L5.Delay

// =============================================================================
// Board-specific configuration
// =============================================================================

/// NUCLEO-L552ZE-Q User LEDs
module LED =
    // LD1 (green) - PC7
    let LD1_PORT = 'C'
    let LD1_PIN = 7

    // LD2 (blue) - PB7
    let LD2_PORT = 'B'
    let LD2_PIN = 7

    // LD3 (red) - PA9
    let LD3_PORT = 'A'
    let LD3_PIN = 9

    /// Initialize all LEDs
    let initAll() =
        configureAsOutput LD1_PORT LD1_PIN
        configureAsOutput LD2_PORT LD2_PIN
        configureAsOutput LD3_PORT LD3_PIN

    /// Turn on LD1 (green)
    let ld1On() = setPin LD1_PORT LD1_PIN

    /// Turn off LD1 (green)
    let ld1Off() = clearPin LD1_PORT LD1_PIN

    /// Toggle LD1 (green)
    let ld1Toggle() = togglePin LD1_PORT LD1_PIN

    /// Turn on LD2 (blue)
    let ld2On() = setPin LD2_PORT LD2_PIN

    /// Turn off LD2 (blue)
    let ld2Off() = clearPin LD2_PORT LD2_PIN

    /// Toggle LD2 (blue)
    let ld2Toggle() = togglePin LD2_PORT LD2_PIN

    /// Turn on LD3 (red)
    let ld3On() = setPin LD3_PORT LD3_PIN

    /// Turn off LD3 (red)
    let ld3Off() = clearPin LD3_PORT LD3_PIN

    /// Toggle LD3 (red)
    let ld3Toggle() = togglePin LD3_PORT LD3_PIN

    /// All LEDs off
    let allOff() =
        ld1Off()
        ld2Off()
        ld3Off()

    /// All LEDs on
    let allOn() =
        ld1On()
        ld2On()
        ld3On()

// =============================================================================
// Blink patterns
// =============================================================================

/// Simple blink - toggle LED at fixed interval
let simpleBlink() =
    LED.initAll()
    LED.allOff()

    while true do
        LED.ld1Toggle()
        ms 500

/// Chase pattern - LEDs light up in sequence
let chaseBlink() =
    LED.initAll()
    LED.allOff()

    while true do
        // LD1 on, others off
        LED.ld1On()
        LED.ld2Off()
        LED.ld3Off()
        ms 200

        // LD2 on, others off
        LED.ld1Off()
        LED.ld2On()
        LED.ld3Off()
        ms 200

        // LD3 on, others off
        LED.ld1Off()
        LED.ld2Off()
        LED.ld3On()
        ms 200

/// Heartbeat pattern - two quick blinks then pause
let heartbeatBlink() =
    LED.initAll()
    LED.allOff()

    while true do
        // First beat
        LED.ld1On()
        ms 100
        LED.ld1Off()
        ms 100

        // Second beat
        LED.ld1On()
        ms 100
        LED.ld1Off()

        // Pause
        ms 600

// =============================================================================
// Entry point
// =============================================================================

/// Main entry point
/// Called by the reset handler after runtime initialization
[<EntryPoint>]
let main argv =
    // Run simple blink pattern
    // Change to chaseBlink() or heartbeatBlink() for other patterns
    simpleBlink()

    // Should never reach here (blink loops forever)
    0
