; Firefly-generated MLIR for STM32L5-Blinky
; Target: thumbv8m.main-none-eabihf
; This is a stub - full code generation not yet implemented

module {
  // Entry point for STM32L5-Blinky
  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }
}