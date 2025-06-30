module Dabbit.Integration.AlloyBindings

open Core.Types.TypeSystem
open Core.Types.Dialects
open Dabbit.Bindings.SymbolRegistry
open Dabbit.Bindings.PatternLibrary

/// Alloy library function signatures and bindings
module AlloyFunctions =
    
    /// Core memory management functions
    let memoryFunctions = [
        "NativePtr.stackalloc", {
            QualifiedName = "Alloy.NativePtr.stackalloc"
            ShortName = "stackalloc"
            ParameterTypes = [MLIRTypes.i32]  // size
            ReturnType = MLIRTypes.memref MLIRTypes.i8
            Operation = DialectOp(MLIRDialect.MemRef, "alloca", Map["element_type", "i8"])
            Namespace = "Alloy.NativePtr"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        
        "NativePtr.alloc", {
            QualifiedName = "Alloy.NativePtr.alloc"
            ShortName = "alloc"
            ParameterTypes = [MLIRTypes.i32]  // size
            ReturnType = MLIRTypes.memref MLIRTypes.i8
            Operation = DialectOp(MLIRDialect.MemRef, "alloc", Map["static", "true"])
            Namespace = "Alloy.NativePtr"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        
        "NativePtr.free", {
            QualifiedName = "Alloy.NativePtr.free"
            ShortName = "free"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.void_
            Operation = DialectOp(MLIRDialect.MemRef, "dealloc", Map.empty)
            Namespace = "Alloy.NativePtr"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        
        "NativePtr.copy", {
            QualifiedName = "Alloy.NativePtr.copy"
            ShortName = "copy"
            ParameterTypes = [
                MLIRTypes.memref MLIRTypes.i8  // source
                MLIRTypes.memref MLIRTypes.i8  // dest
                MLIRTypes.i32                   // count
            ]
            ReturnType = MLIRTypes.void_
            Operation = DialectOp(MLIRDialect.MemRef, "copy", Map.empty)
            Namespace = "Alloy.NativePtr"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
    ]
    
    /// Console I/O functions
    let consoleFunctions = [
        "Console.write", {
            QualifiedName = "Alloy.IO.Console.write"
            ShortName = "write"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.void_
            Operation = ExternalCall("printf", Some "libc")
            Namespace = "Alloy.IO.Console"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
        
        "Console.writeLine", {
            QualifiedName = "Alloy.IO.Console.writeLine"
            ShortName = "writeLine"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.void_
            Operation = ExternalCall("puts", Some "libc")
            Namespace = "Alloy.IO.Console"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
        
        "Console.read", {
            QualifiedName = "Alloy.IO.Console.read"
            ShortName = "read"
            ParameterTypes = [
                MLIRTypes.memref MLIRTypes.i8  // buffer
                MLIRTypes.i32                   // max size
            ]
            ReturnType = MLIRTypes.i32  // bytes read
            Operation = ExternalCall("fgets", Some "libc")
            Namespace = "Alloy.IO.Console"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
    ]
    
    /// String operations (zero-allocation)
    let stringFunctions = [
        "String.length", {
            QualifiedName = "Alloy.String.length"
            ShortName = "length"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.i32
            Operation = ExternalCall("strlen", Some "libc")
            Namespace = "Alloy.String"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
        
        "String.copy", {
            QualifiedName = "Alloy.String.copy"
            ShortName = "copy"
            ParameterTypes = [
                MLIRTypes.memref MLIRTypes.i8  // source
                MLIRTypes.memref MLIRTypes.i8  // dest
                MLIRTypes.i32                   // max chars
            ]
            ReturnType = MLIRTypes.i32  // chars copied
            Operation = Transform("string_copy", ["bounds_checked"])
            Namespace = "Alloy.String"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        
        "String.compare", {
            QualifiedName = "Alloy.String.compare"
            ShortName = "compare"
            ParameterTypes = [
                MLIRTypes.memref MLIRTypes.i8  // str1
                MLIRTypes.memref MLIRTypes.i8  // str2
            ]
            ReturnType = MLIRTypes.i32  // -1, 0, 1
            Operation = ExternalCall("strcmp", Some "libc")
            Namespace = "Alloy.String"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
    ]
    
    /// Buffer operations
    let bufferFunctions = [
        "Buffer.create", {
            QualifiedName = "Alloy.Buffer.create"
            ShortName = "create"
            ParameterTypes = [MLIRTypes.i32]  // size
            ReturnType = MLIRTypes.memref MLIRTypes.i8
            Operation = DialectOp(MLIRDialect.MemRef, "alloca", Map["allocation", "stack"])
            Namespace = "Alloy.Buffer"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        
        "Buffer.slice", {
            QualifiedName = "Alloy.Buffer.slice"
            ShortName = "slice"
            ParameterTypes = [
                MLIRTypes.memref MLIRTypes.i8  // buffer
                MLIRTypes.i32                   // start
                MLIRTypes.i32                   // length
            ]
            ReturnType = MLIRTypes.memref MLIRTypes.i8
            Operation = DialectOp(MLIRDialect.MemRef, "subview", Map["view", "true"])
            Namespace = "Alloy.Buffer"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        
        "Buffer.asSpan", {
            QualifiedName = "Alloy.Buffer.asSpan"
            ShortName = "asSpan"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.memref MLIRTypes.i8  // Same type, different view
            Operation = Transform("buffer_to_span", ["zero_cost"])
            Namespace = "Alloy.Buffer"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
    ]
    
    /// Math operations (zero-allocation)
    let mathFunctions = [
        "Math.min", {
            QualifiedName = "Alloy.Math.min"
            ShortName = "min"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            Operation = DialectOp(MLIRDialect.Arith, "minsi", Map.empty)
            Namespace = "Alloy.Math"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        
        "Math.max", {
            QualifiedName = "Alloy.Math.max"
            ShortName = "max"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            Operation = DialectOp(MLIRDialect.Arith, "maxsi", Map.empty)
            Namespace = "Alloy.Math"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        
        "Math.abs", {
            QualifiedName = "Alloy.Math.abs"
            ShortName = "abs"
            ParameterTypes = [MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            Operation = Transform("abs_i32", ["pure"])
            Namespace = "Alloy.Math"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
    ]

/// Register all Alloy functions with the symbol registry
let registerAlloySymbols (registry: SymbolRegistry) : SymbolRegistry =
    let allFunctions = 
        AlloyFunctions.memoryFunctions @
        AlloyFunctions.consoleFunctions @
        AlloyFunctions.stringFunctions @
        AlloyFunctions.bufferFunctions @
        AlloyFunctions.mathFunctions
    
    allFunctions 
    |> List.fold (fun reg (shortName, symbol) ->
        // Register symbol in the registry
        let updatedState = SymbolResolution.registerSymbol symbol reg.State
        { reg with State = updatedState }
    ) registry

/// MLIR lowering patterns for Alloy functions
module AlloyPatterns =
    open Dabbit.CodeGeneration.MLIREmitter
    
    /// Pattern for stack allocation
    let stackAllocPattern (size: int) : MLIRBuilder<string> =
        mlir {
            let! bufferSSA = nextSSA "stack_buffer"
            do! emitLine (sprintf "%s = memref.alloca() : memref<%dxi8>" bufferSSA size)
            return bufferSSA
        }
    
    /// Pattern for console read with buffer
    let consoleReadPattern (bufferSSA: string) (sizeSSA: string) : MLIRBuilder<string> =
        mlir {
            do! requireExternal "fgets"
            do! requireExternal "stdin"
            
            let! stdinSSA = nextSSA "stdin_ptr"
            do! emitLine (sprintf "%s = llvm.mlir.addressof @stdin : !llvm.ptr" stdinSSA)
            
            let! stdinLoadSSA = nextSSA "stdin"
            do! emitLine (sprintf "%s = llvm.load %s : !llvm.ptr -> !llvm.ptr" stdinLoadSSA stdinSSA)
            
            let! fgetsResult = nextSSA "fgets_result"
            do! emitLine (sprintf "%s = func.call @fgets(%s, %s, %s) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr" 
                fgetsResult bufferSSA sizeSSA stdinLoadSSA)
            
            // Calculate actual length read
            do! requireExternal "strlen"
            let! lengthSSA = nextSSA "read_length"
            do! emitLine (sprintf "%s = func.call @strlen(%s) : (!llvm.ptr) -> i32" lengthSSA bufferSSA)
            
            return lengthSSA
        }
    
    /// Pattern for bounds-checked array access
    let boundsCheckPattern (arraySSA: string) (indexSSA: string) (sizeSSA: string) : MLIRBuilder<string> =
        mlir {
            // Generate bounds check
            let! inBoundsSSA = nextSSA "in_bounds"
            do! emitLine (sprintf "%s = arith.cmpi ult, %s, %s : i32" inBoundsSSA indexSSA sizeSSA)
            
            // Assert bounds (in debug mode)
            do! emitLine (sprintf "assert %s, \"Array index out of bounds\"" inBoundsSSA)
            
            // Perform load
            let! resultSSA = nextSSA "elem"
            do! emitLine (sprintf "%s = memref.load %s[%s] : memref<?xi8>" resultSSA arraySSA indexSSA)
            
            return resultSSA
        }

/// Type mappings for Alloy types to MLIR
module AlloyTypes =
    /// StackBuffer<'T> maps to memref
    let stackBufferType (elementType: MLIRType) = MLIRTypes.memref elementType
    
    /// Span<'T> maps to memref (view)
    let spanType (elementType: MLIRType) = MLIRTypes.memref elementType
    
    /// NativePtr<'T> maps to LLVM pointer
    let nativePtrType (elementType: MLIRType) = 
        { MLIRTypes.memref elementType with Category = MLIRTypeCategory.MemRef }

/// Verification helpers for Alloy constraints
module AlloyVerification =
    /// Verify zero-allocation guarantee
    let verifyZeroHeapAllocation (mlirModule: string) : Result<unit, string> =
        if mlirModule.Contains("memref.alloc(") && not (mlirModule.Contains("memref.alloca(")) then
            Error "Heap allocation detected - violates zero-allocation guarantee"
        elif mlirModule.Contains("malloc") || mlirModule.Contains("new") then
            Error "Dynamic allocation detected - violates zero-allocation guarantee"
        else
            Ok ()
    
    /// Verify bounded stack usage
    let verifyBoundedStack (mlirModule: string) (maxStackSize: int) : Result<unit, string> =
        // Simple heuristic - count alloca sizes
        let allocaPattern = System.Text.RegularExpressions.Regex(@"memref\.alloca\(\) : memref<(\d+)xi8>")
        let totalSize = 
            allocaPattern.Matches(mlirModule)
            |> Seq.cast<System.Text.RegularExpressions.Match>
            |> Seq.sumBy (fun m -> int m.Groups.[1].Value)
        
        if totalSize > maxStackSize then
            Error (sprintf "Stack usage %d exceeds maximum %d" totalSize maxStackSize)
        else
            Ok ()
    
    /// Verify no hidden allocations
    let verifyNoHiddenAllocations (mlirModule: string) : Result<unit, string> =
        let suspiciousPatterns = [
            "func.call @malloc"
            "func.call @calloc"
            "llvm.call @_Znwm"  // C++ new
            "llvm.call @GC_"    // GC calls
        ]
        
        match suspiciousPatterns |> List.tryFind (fun p -> mlirModule.Contains(p)) with
        | Some pattern -> Error (sprintf "Hidden allocation detected: %s" pattern)
        | None -> Ok ()