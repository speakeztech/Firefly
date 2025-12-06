/// CortexA53 - ARM Cortex-A53 startup code for ARM64 SBCs
/// Provides minimal bare-metal startup for boards like Libre Sweet Potato
module Embedded.Startup.CortexA53

open Alloy.Memory

// =============================================================================
// External symbols (provided by linker script)
// =============================================================================

/// End of stack
[<Extern>]
extern val __stack_top: nativeint

/// Start of .data section in RAM
[<Extern>]
extern val __data_start: nativeint

/// End of .data section in RAM
[<Extern>]
extern val __data_end: nativeint

/// Load address of .data section
[<Extern>]
extern val __data_load: nativeint

/// Start of .bss section
[<Extern>]
extern val __bss_start: nativeint

/// End of .bss section
[<Extern>]
extern val __bss_end: nativeint

// =============================================================================
// Exception vectors (AArch64)
// =============================================================================

/// Synchronous exception handler
[<Naked; NoInline>]
let syncHandler() =
    while true do ()

/// IRQ handler
[<Naked; NoInline>]
let irqHandler() =
    while true do ()

/// FIQ handler
[<Naked; NoInline>]
let fiqHandler() =
    while true do ()

/// SError handler
[<Naked; NoInline>]
let serrorHandler() =
    while true do ()

// =============================================================================
// Runtime initialization
// =============================================================================

/// Copy initialized data
let private copyDataSection() =
    let mutable src = __data_load
    let mutable dst = __data_start
    while dst < __data_end do
        Ptr.write<byte> dst (Ptr.read<byte> src)
        src <- src + 1n
        dst <- dst + 1n

/// Zero BSS section
let private zeroBssSection() =
    let mutable ptr = __bss_start
    while ptr < __bss_end do
        Ptr.write<byte> ptr 0uy
        ptr <- ptr + 1n

/// Initialize MMU (if needed for bare-metal)
let private initMMU() =
    // For simple bare-metal, we might run with MMU disabled
    // or with identity mapping
    ()

/// Initialize caches
let private initCaches() =
    // Enable instruction and data caches for performance
    ()

// =============================================================================
// Entry point
// =============================================================================

/// Primary entry point after bootloader
[<Naked; NoInline>]
let _start() =
    // Initialize stack pointer
    // (Usually done by bootloader, but we set it explicitly)

    // Copy .data section
    copyDataSection()

    // Zero .bss section
    zeroBssSection()

    // Initialize MMU if needed
    initMMU()

    // Enable caches
    initCaches()

    // Call main
    let result = main [||]

    // If main returns, loop forever
    while true do
        ()

// =============================================================================
// Exception vector table (AArch64)
// =============================================================================

/// AArch64 exception vector table
/// Each vector is 0x80 (128) bytes apart
[<VectorTable(Address = 0x40000000u, Alignment = 2048)>]
let exceptionVectors = [|
    // Current EL with SP0
    syncHandler     // Synchronous
    irqHandler      // IRQ
    fiqHandler      // FIQ
    serrorHandler   // SError

    // Current EL with SPx
    syncHandler
    irqHandler
    fiqHandler
    serrorHandler

    // Lower EL using AArch64
    syncHandler
    irqHandler
    fiqHandler
    serrorHandler

    // Lower EL using AArch32
    syncHandler
    irqHandler
    fiqHandler
    serrorHandler
|]

// =============================================================================
// Main function declaration
// =============================================================================

[<Extern>]
extern val main: string[] -> int
