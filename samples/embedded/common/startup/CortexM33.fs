/// CortexM33 - ARM Cortex-M33 startup code for STM32L5 series
/// Provides vector table, reset handler, and runtime initialization
module Embedded.Startup.CortexM33

open Alloy.Memory

// =============================================================================
// External symbols (provided by linker script)
// =============================================================================

/// End of stack (top of RAM)
[<Extern>]
extern val __stack_top: nativeint

/// Start of .data section in RAM
[<Extern>]
extern val __data_start: nativeint

/// End of .data section in RAM
[<Extern>]
extern val __data_end: nativeint

/// Load address of .data section in Flash
[<Extern>]
extern val __data_load: nativeint

/// Start of .bss section
[<Extern>]
extern val __bss_start: nativeint

/// End of .bss section
[<Extern>]
extern val __bss_end: nativeint

// =============================================================================
// Default exception handlers
// =============================================================================

/// Default handler for unused interrupts - infinite loop
[<Naked; NoInline>]
let defaultHandler() =
    while true do
        // WFI - Wait For Interrupt (low power)
        ()

/// Hard fault handler - indicates serious error
[<Naked; NoInline>]
let hardFaultHandler() =
    // In a real application, this would:
    // - Save fault information
    // - Blink an error LED pattern
    // - Attempt recovery or safe shutdown
    while true do
        ()

/// Memory management fault handler
[<Naked; NoInline>]
let memManageHandler() =
    while true do
        ()

/// Bus fault handler
[<Naked; NoInline>]
let busFaultHandler() =
    while true do
        ()

/// Usage fault handler (undefined instruction, etc.)
[<Naked; NoInline>]
let usageFaultHandler() =
    while true do
        ()

/// Secure fault handler (TrustZone violations)
[<Naked; NoInline>]
let secureFaultHandler() =
    while true do
        ()

// =============================================================================
// Runtime initialization
// =============================================================================

/// Copy initialized data from Flash to RAM
let private copyDataSection() =
    let mutable src = __data_load
    let mutable dst = __data_start
    while dst < __data_end do
        Ptr.write<byte> dst (Ptr.read<byte> src)
        src <- src + 1n
        dst <- dst + 1n

/// Zero the BSS section
let private zeroBssSection() =
    let mutable ptr = __bss_start
    while ptr < __bss_end do
        Ptr.write<byte> ptr 0uy
        ptr <- ptr + 1n

/// Initialize the C runtime (call static constructors if any)
let private initRuntime() =
    // Firefly doesn't use static constructors, but this is where
    // they would be called if needed
    ()

// =============================================================================
// Reset handler - entry point after reset
// =============================================================================

/// Reset handler - called on system reset
/// This is the true entry point of the application
[<Naked; NoInline>]
let resetHandler() =
    // Copy .data section from Flash to RAM
    copyDataSection()

    // Zero .bss section
    zeroBssSection()

    // Initialize runtime
    initRuntime()

    // Call main application entry point
    // The main function should be defined in the application code
    let result = main [||]

    // If main returns, loop forever
    // (embedded applications typically never return from main)
    while true do
        ()

// =============================================================================
// Vector table
// =============================================================================

/// ARM Cortex-M33 vector table
/// Must be placed at address 0x00000000 (or VTOR offset)
[<VectorTable(Address = 0x08000000u)>]
let vectorTable = [|
    // Stack pointer initial value
    __stack_top

    // Reset handler (entry point)
    resetHandler

    // System exceptions
    defaultHandler      // NMI
    hardFaultHandler    // Hard Fault
    memManageHandler    // Memory Management Fault
    busFaultHandler     // Bus Fault
    usageFaultHandler   // Usage Fault
    secureFaultHandler  // Secure Fault (ARMv8-M)
    defaultHandler      // Reserved
    defaultHandler      // Reserved
    defaultHandler      // Reserved
    defaultHandler      // SVCall
    defaultHandler      // Debug Monitor
    defaultHandler      // Reserved
    defaultHandler      // PendSV
    defaultHandler      // SysTick

    // External interrupts (STM32L5-specific)
    // These would be filled in based on the specific STM32L5 variant
    // For now, use default handlers

    // IRQ 0-15: Window Watchdog, PVD, RTC Tamper, RTC Wakeup, Flash, RCC, EXTI0-4
    defaultHandler; defaultHandler; defaultHandler; defaultHandler
    defaultHandler; defaultHandler; defaultHandler; defaultHandler
    defaultHandler; defaultHandler; defaultHandler; defaultHandler
    defaultHandler; defaultHandler; defaultHandler; defaultHandler

    // IRQ 16-31: EXTI5-9, TIM1, TIM2, etc.
    defaultHandler; defaultHandler; defaultHandler; defaultHandler
    defaultHandler; defaultHandler; defaultHandler; defaultHandler
    defaultHandler; defaultHandler; defaultHandler; defaultHandler
    defaultHandler; defaultHandler; defaultHandler; defaultHandler

    // IRQ 32-47: I2C, SPI, USART, etc.
    defaultHandler; defaultHandler; defaultHandler; defaultHandler
    defaultHandler; defaultHandler; defaultHandler; defaultHandler
    defaultHandler; defaultHandler; defaultHandler; defaultHandler
    defaultHandler; defaultHandler; defaultHandler; defaultHandler

    // IRQ 48-63: DMA, ADC, etc.
    defaultHandler; defaultHandler; defaultHandler; defaultHandler
    defaultHandler; defaultHandler; defaultHandler; defaultHandler
    defaultHandler; defaultHandler; defaultHandler; defaultHandler
    defaultHandler; defaultHandler; defaultHandler; defaultHandler

    // Additional IRQs up to 108 for STM32L5
    // ... (truncated for brevity, would continue to IRQ 108)
|]

// =============================================================================
// Main function declaration (implemented in application)
// =============================================================================

/// Main function signature - must be implemented by the application
[<Extern>]
extern val main: string[] -> int
