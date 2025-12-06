/*
 * STM32L552 Linker Script
 * For NUCLEO-L552ZE-Q development board
 *
 * Memory Layout:
 * - Flash: 512KB at 0x08000000
 * - SRAM1: 192KB at 0x20000000
 * - SRAM2: 64KB at 0x20030000
 *
 * This script is designed for non-secure firmware.
 * For TrustZone configurations, adjust memory regions accordingly.
 */

/* Entry point */
ENTRY(resetHandler)

/* Memory regions */
MEMORY
{
    /* Main Flash memory */
    FLASH (rx)   : ORIGIN = 0x08000000, LENGTH = 512K

    /* SRAM1 - Primary RAM */
    SRAM1 (rwx)  : ORIGIN = 0x20000000, LENGTH = 192K

    /* SRAM2 - Secondary RAM (can be retained in low-power modes) */
    SRAM2 (rwx)  : ORIGIN = 0x20030000, LENGTH = 64K
}

/* Stack and heap sizes */
__stack_size = 8K;
__heap_size = 0;  /* Firefly uses stack-only allocation by default */

/* Calculate stack top */
__stack_top = ORIGIN(SRAM1) + LENGTH(SRAM1);

SECTIONS
{
    /* Vector table and code - must start at Flash origin */
    .text :
    {
        . = ALIGN(4);
        __text_start = .;

        /* Vector table at the very beginning */
        KEEP(*(.isr_vector))
        KEEP(*(.vectors))

        . = ALIGN(4);

        /* Code sections */
        *(.text)
        *(.text*)

        /* Read-only data */
        *(.rodata)
        *(.rodata*)

        /* ARM exception handling tables */
        *(.ARM.extab*)
        *(.ARM.exidx*)

        . = ALIGN(4);
        __text_end = .;
    } > FLASH

    /* ARM unwinding information */
    .ARM.extab :
    {
        *(.ARM.extab* .gnu.linkonce.armextab.*)
    } > FLASH

    .ARM.exidx :
    {
        __exidx_start = .;
        *(.ARM.exidx* .gnu.linkonce.armexidx.*)
        __exidx_end = .;
    } > FLASH

    /* Initialized data - loaded from Flash, copied to RAM at startup */
    .data :
    {
        . = ALIGN(4);
        __data_start = .;

        *(.data)
        *(.data*)

        . = ALIGN(4);
        __data_end = .;
    } > SRAM1 AT > FLASH

    /* Load address for .data section (in Flash) */
    __data_load = LOADADDR(.data);

    /* Uninitialized data - zeroed at startup */
    .bss :
    {
        . = ALIGN(4);
        __bss_start = .;

        *(.bss)
        *(.bss*)
        *(COMMON)

        . = ALIGN(4);
        __bss_end = .;
    } > SRAM1

    /* Heap (if used) */
    .heap :
    {
        . = ALIGN(8);
        __heap_start = .;
        . = . + __heap_size;
        __heap_end = .;
    } > SRAM1

    /* Stack */
    .stack :
    {
        . = ALIGN(8);
        __stack_start = .;
        . = . + __stack_size;
        /* __stack_top defined above as end of SRAM1 */
    } > SRAM1

    /* SRAM2 section - for data that should be in SRAM2 */
    .sram2 (NOLOAD) :
    {
        . = ALIGN(4);
        __sram2_start = .;
        *(.sram2)
        *(.sram2*)
        . = ALIGN(4);
        __sram2_end = .;
    } > SRAM2

    /* Discard sections we don't need */
    /DISCARD/ :
    {
        *(.comment)
        *(.note*)
    }
}

/* Provide symbols for startup code */
PROVIDE(__stack_top = __stack_top);
PROVIDE(__data_start = __data_start);
PROVIDE(__data_end = __data_end);
PROVIDE(__data_load = __data_load);
PROVIDE(__bss_start = __bss_start);
PROVIDE(__bss_end = __bss_end);
PROVIDE(__heap_start = __heap_start);
PROVIDE(__heap_end = __heap_end);

/* Assert that everything fits */
ASSERT(__stack_top <= ORIGIN(SRAM1) + LENGTH(SRAM1), "Stack overflow")
ASSERT(__bss_end <= __stack_start, "BSS overlaps with stack")
