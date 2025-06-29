module Dabbit.Integration.AlloyBindings

open Core.Types.TypeSystem
open Dabbit.Bindings.SymbolRegistry
open Dabbit.Bindings.PatternLibrary

/// Alloy library function signatures and bindings
module AlloyFunctions =
    
    /// Core memory management functions
    let memoryFunctions = [
        "NativePtr.stackalloc", {
            QualifiedName = "Alloy.NativePtr.stackalloc"
            ParameterTypes = [MLIRTypes.i32]  // size
            ReturnType = MLIRTypes.memref MLIRTypes.i8
            SymbolKind = Function
            Attributes = Map.ofList ["allocation", "stack"; "safety", "bounded"]
        }
        
        "NativePtr.alloc", {
            QualifiedName = "Alloy.NativePtr.alloc"
            ParameterTypes = [MLIRTypes.i32]  // size
            ReturnType = MLIRTypes.memref MLIRTypes.i8
            SymbolKind = Function
            Attributes = Map.ofList ["allocation", "static"; "safety", "checked"]
        }
        
        "NativePtr.free", {
            QualifiedName = "Alloy.NativePtr.free"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.void_
            SymbolKind = Function
            Attributes = Map.ofList ["deallocation", "explicit"]
        }
        
        "NativePtr.copy", {
            QualifiedName = "Alloy.NativePtr.copy"
            ParameterTypes = [
                MLIRTypes.memref MLIRTypes.i8  // source
                MLIRTypes.memref MLIRTypes.i8  // dest
                MLIRTypes.i32                   // count
            ]
            ReturnType = MLIRTypes.void_
            SymbolKind = Function
            Attributes = Map.ofList ["operation", "memcpy"]
        }
    ]
    
    /// Console I/O functions
    let consoleFunctions = [
        "Console.readLine", {
            QualifiedName = "Alloy.Console.readLine"
            ParameterTypes = [
                MLIRTypes.memref MLIRTypes.i8  // buffer
                MLIRTypes.i32                   // maxLength
            ]
            ReturnType = MLIRTypes.i32  // actual length read
            SymbolKind = Function
            Attributes = Map.ofList ["io", "console"; "blocking", "true"]
        }
        
        "Console.write", {
            QualifiedName = "Alloy.Console.write"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]  // string buffer
            ReturnType = MLIRTypes.void_
            SymbolKind = Function
            Attributes = Map.ofList ["io", "console"]
        }
        
        "Console.writeLine", {
            QualifiedName = "Alloy.Console.writeLine"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]  // string buffer
            ReturnType = MLIRTypes.void_
            SymbolKind = Function
            Attributes = Map.ofList ["io", "console"]
        }
    ]
    
    /// String manipulation functions
    let stringFunctions = [
        "String.length", {
            QualifiedName = "Alloy.String.length"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.i32
            SymbolKind = Function
            Attributes = Map.ofList ["pure", "true"]
        }
        
        "String.copy", {
            QualifiedName = "Alloy.String.copy"
            ParameterTypes = [
                MLIRTypes.memref MLIRTypes.i8  // source
                MLIRTypes.memref MLIRTypes.i8  // dest
                MLIRTypes.i32                   // maxLength
            ]
            ReturnType = MLIRTypes.i32  // actual copied
            SymbolKind = Function
            Attributes = Map.ofList ["bounds", "checked"]
        }
        
        "String.compare", {
            QualifiedName = "Alloy.String.compare"
            ParameterTypes = [
                MLIRTypes.memref MLIRTypes.i8  // str1
                MLIRTypes.memref MLIRTypes.i8  // str2
            ]
            ReturnType = MLIRTypes.i32  // -1, 0, 1
            SymbolKind = Function
            Attributes = Map.ofList ["pure", "true"]
        }
    ]
    
    /// Buffer operations
    let bufferFunctions = [
        "Buffer.create", {
            QualifiedName = "Alloy.Buffer.create"
            ParameterTypes = [MLIRTypes.i32]  // size
            ReturnType = MLIRTypes.memref MLIRTypes.i8
            SymbolKind = Function
            Attributes = Map.ofList ["allocation", "stack"]
        }
        
        "Buffer.slice", {
            QualifiedName = "Alloy.Buffer.slice"
            ParameterTypes = [
                MLIRTypes.memref MLIRTypes.i8  // buffer
                MLIRTypes.i32                   // start
                MLIRTypes.i32                   // length
            ]
            ReturnType = MLIRTypes.memref MLIRTypes.i8
            SymbolKind = Function
            Attributes = Map.ofList ["view", "true"; "allocation", "none"]
        }
        
        "Buffer.asSpan", {
            QualifiedName = "Alloy.Buffer.asSpan"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.memref MLIRTypes.i8  // Same type, different view
            SymbolKind = Function
            Attributes = Map.ofList ["view", "true"; "zero-cost", "true"]
        }
    ]
    
    /// Math operations (zero-allocation)
    let mathFunctions = [
        "Math.min", {
            QualifiedName = "Alloy.Math.min"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            SymbolKind = Function
            Attributes = Map.ofList ["pure", "true"; "inline", "always"]
        }
        
        "Math.max", {
            QualifiedName = "Alloy.Math.max"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            SymbolKind = Function
            Attributes = Map.ofList ["pure", "true"; "inline", "always"]
        }
        
        "Math.abs", {
            QualifiedName = "Alloy.Math.abs"
            ParameterTypes = [MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            SymbolKind = Function
            Attributes = Map.ofList ["pure", "true"; "inline", "always"]
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
        // Register with both short and qualified names
        let reg' = Operations.registerSymbol symbol reg
        Operations.registerSymbolAlias shortName symbol.QualifiedName reg'
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
    let boundsCheckedLoadPattern (arraySSA: string) (indexSSA: string) (sizeSSA: string) : MLIRBuilder<string> =
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

/// Integration with pattern library
let registerAlloyPatterns (library: PatternLibrary) : PatternLibrary =
    library
    |> PatternLibrary.registerPattern 
        "alloy.stackalloc"
        (PatternNode("StackAlloc", 
            [PatternCapture("size", PatternWildcard)],
            Map.ofList ["allocation", "stack"]))
    |> PatternLibrary.registerPattern
        "alloy.console.read"
        (PatternNode("ConsoleRead",
            [PatternCapture("buffer", PatternWildcard);
             PatternCapture("size", PatternWildcard)],
            Map.ofList ["io", "console"; "blocking", "true"]))
    |> PatternLibrary.registerPattern
        "alloy.bounds.check"
        (PatternNode("BoundsCheck",
            [PatternCapture("array", PatternWildcard);
             PatternCapture("index", PatternWildcard);
             PatternCapture("size", PatternWildcard)],
            Map.ofList ["safety", "bounds"; "debug", "assert"]))

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