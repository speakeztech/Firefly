/*
 * Allwinner H6 Linker Script
 * For Libre Sweet Potato and similar H6-based boards
 *
 * Memory Layout (bare-metal):
 * - SRAM A1: 32KB at 0x00020000 (fast, always-on)
 * - SRAM A2: 32KB at 0x00028000
 * - SRAM C: 128KB at 0x00040000
 * - DDR: 2GB+ at 0x40000000 (requires DDR initialization)
 *
 * This script assumes U-Boot or another bootloader has:
 * - Initialized DDR
 * - Loaded our code to DDR
 * - Jumped to our entry point
 */

/* Entry point */
ENTRY(_start)

/* Memory regions */
MEMORY
{
    /* On-chip SRAM - available without DDR init */
    SRAM_A1 (rwx) : ORIGIN = 0x00020000, LENGTH = 32K
    SRAM_A2 (rwx) : ORIGIN = 0x00028000, LENGTH = 32K
    SRAM_C  (rwx) : ORIGIN = 0x00040000, LENGTH = 128K

    /* DDR memory - main execution region */
    /* Assumes bootloader loads us here */
    DDR (rwx)     : ORIGIN = 0x40000000, LENGTH = 64M
}

/* Stack and heap sizes */
__stack_size = 64K;
__heap_size = 0;  /* Stack-only by default */

SECTIONS
{
    /* Exception vector table - must be 2KB aligned for AArch64 */
    .vectors :
    {
        . = ALIGN(2048);
        __vectors_start = .;
        KEEP(*(.vectors))
        . = ALIGN(2048);
        __vectors_end = .;
    } > DDR

    /* Code */
    .text :
    {
        . = ALIGN(8);
        __text_start = .;

        *(.text.boot)   /* Boot code first */
        *(.text._start) /* Entry point */
        *(.text)
        *(.text*)

        . = ALIGN(8);
        __text_end = .;
    } > DDR

    /* Read-only data */
    .rodata :
    {
        . = ALIGN(8);
        __rodata_start = .;

        *(.rodata)
        *(.rodata*)

        . = ALIGN(8);
        __rodata_end = .;
    } > DDR

    /* Initialized data */
    .data :
    {
        . = ALIGN(8);
        __data_start = .;

        *(.data)
        *(.data*)

        . = ALIGN(8);
        __data_end = .;
    } > DDR

    /* For bare-metal from bootloader, data is already in place */
    __data_load = __data_start;

    /* Uninitialized data */
    .bss :
    {
        . = ALIGN(8);
        __bss_start = .;

        *(.bss)
        *(.bss*)
        *(COMMON)

        . = ALIGN(8);
        __bss_end = .;
    } > DDR

    /* Stack */
    .stack :
    {
        . = ALIGN(16);  /* AArch64 requires 16-byte stack alignment */
        __stack_start = .;
        . = . + __stack_size;
        __stack_top = .;
    } > DDR

    /* Heap (if used) */
    .heap :
    {
        . = ALIGN(8);
        __heap_start = .;
        . = . + __heap_size;
        __heap_end = .;
    } > DDR

    /* SRAM sections for performance-critical code/data */
    .sram_code (NOLOAD) :
    {
        . = ALIGN(4);
        *(.sram_code)
        *(.sram_code*)
    } > SRAM_A1

    .sram_data (NOLOAD) :
    {
        . = ALIGN(4);
        *(.sram_data)
        *(.sram_data*)
    } > SRAM_A2

    /* Discard */
    /DISCARD/ :
    {
        *(.comment)
        *(.note*)
    }
}

/* Symbols for startup code */
PROVIDE(__stack_top = __stack_top);
PROVIDE(__data_start = __data_start);
PROVIDE(__data_end = __data_end);
PROVIDE(__data_load = __data_load);
PROVIDE(__bss_start = __bss_start);
PROVIDE(__bss_end = __bss_end);
