module Dabbit.CodeGeneration.MLIRBuiltins

open Core.Types.TypeSystem
open Core.XParsec.Foundation
open MLIRSyntax
open MLIREmitter

/// Built-in function registry using XParsec patterns
type BuiltinSignature = {
    Name: string
    ParameterTypes: MLIRType list
    ReturnType: MLIRType
    Implementation: MLIRValue list -> MLIRCombinator<MLIRValue>
}

/// Built-in function catalog using XParsec combinators
module Catalog =
    
    /// String operations using XParsec combinators
    let stringConcat (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] ->
                let! result = nextSSA "concat"
                do! requireExternal "strcat"
                
                // Use XParsec to build the operation
                let funcCallOp = buildFuncCall result "strcat" 
                                    [left.SSA; right.SSA] 
                                    "(!llvm.ptr, !llvm.ptr) -> !llvm.ptr"
                
                do! emitLine funcCallOp
                return Core.createValue result MLIRTypes.string_
            | _ -> 
                return! fail "string_concat" "Expected exactly 2 arguments"
        }
    
    let stringLength (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [str] ->
                let! result = nextSSA "strlen"
                do! requireExternal "strlen"
                
                let funcCallOp = buildFuncCall result "strlen" [str.SSA] "(!llvm.ptr) -> i32"
                do! emitLine funcCallOp
                
                return Core.createValue result MLIRTypes.i32
            | _ ->
                return! fail "string_length" "Expected exactly 1 argument"
        }
    
    /// Numeric operations using XParsec builders
    let mathMin (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] ->
                let! cmpResult = nextSSA "cmp"
                let! minResult = nextSSA "min"
                
                // Build comparison operation
                let cmpOp = buildArithOp "cmpi" cmpResult left.SSA right.SSA 
                               (sprintf "slt, %s" (Core.formatType left.Type))
                do! emitLine cmpOp
                
                // Build select operation
                let selectOp = buildSelect minResult cmpResult left.SSA right.SSA 
                                  (Core.formatType left.Type)
                do! emitLine selectOp
                
                return Core.createValue minResult left.Type
            | _ ->
                return! fail "math_min" "Expected exactly 2 arguments"
        }
    
    let mathAbs (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [value] ->
                let! zero = Constants.intConstant 0 32
                let! cmpResult = nextSSA "is_neg"
                let! negResult = nextSSA "neg"
                let! absResult = nextSSA "abs"
                
                // Compare with zero
                let cmpOp = buildArithOp "cmpi" cmpResult value.SSA zero.SSA 
                               (sprintf "slt, %s" (Core.formatType value.Type))
                do! emitLine cmpOp
                
                // Negate value
                let negOp = buildArithOp "subi" negResult zero.SSA value.SSA 
                               (Core.formatType value.Type)
                do! emitLine negOp
                
                // Select positive value
                let selectOp = buildSelect absResult cmpResult negResult value.SSA 
                                  (Core.formatType value.Type)
                do! emitLine selectOp
                
                return Core.createValue absResult value.Type
            | _ ->
                return! fail "math_abs" "Expected exactly 1 argument"
        }
    
    /// Array operations using XParsec patterns
    let arrayCreate (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [size; initValue] ->
                let! arrayPtr = nextSSA "array"
                
                // Build memref allocation
                let allocaOp = buildAlloca arrayPtr (Core.formatType initValue.Type) (Some size.SSA)
                do! emitLine allocaOp
                
                // Initialize elements using scf.for loop
                do! emitLine "// Array initialization loop"
                let! idx = nextSSA "idx"
                let! c0 = Constants.intConstant 0 32
                let! c1 = Constants.intConstant 1 32
                
                // Build SCF for loop structure
                do! emitLine (sprintf "scf.for %s = %s to %s step %s {" idx c0.SSA size.SSA c1.SSA)
                do! emitLine (sprintf "  memref.store %s, %s[%s] : memref<?x%s>" 
                    initValue.SSA arrayPtr idx (Core.formatType initValue.Type))
                do! emitLine "}"
                
                return Core.createValue arrayPtr (MLIRTypes.memref initValue.Type)
            | _ ->
                return! fail "array_create" "Expected exactly 2 arguments (size, initial_value)"
        }
    
    let arrayGet (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [array; index] ->
                let! result = nextSSA "elem"
                
                let loadOp = buildLoad result array.SSA [index.SSA]
                do! emitLine loadOp
                
                // Extract element type from array type
                let elemType = match array.Type with
                               | MemRef et -> et
                               | _ -> MLIRTypes.i32  // Default fallback
                
                return Core.createValue result elemType
            | _ ->
                return! fail "array_get" "Expected exactly 2 arguments (array, index)"
        }
    
    let arraySet (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [array; index; value] ->
                let storeOp = buildStore value.SSA array.SSA [index.SSA]
                do! emitLine storeOp
                
                return! Constants.unitConstant
            | _ ->
                return! fail "array_set" "Expected exactly 3 arguments (array, index, value)"
        }

/// Registry with XParsec-based implementations
module Registry =
    
    /// Helper to create array zero-initialization
    let arrayZeroCreate (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [size] ->
                let! zero = Constants.intConstant 0 32
                return! Catalog.arrayCreate [size; zero]
            | _ ->
                return! fail "Array.zeroCreate" "Expected exactly 1 argument"
        }
    
    /// All built-in functions with XParsec-based implementations
    let builtinFunctions: Map<string, BuiltinSignature> = 
        [
            // String operations
            ("concat", { 
                Name = "concat"
                ParameterTypes = [MLIRTypes.string_; MLIRTypes.string_]
                ReturnType = MLIRTypes.string_
                Implementation = Catalog.stringConcat
            })
            
            ("length", { 
                Name = "length"
                ParameterTypes = [MLIRTypes.string_]
                ReturnType = MLIRTypes.i32
                Implementation = Catalog.stringLength
            })
            
            // Math operations
            ("min", { 
                Name = "min"
                ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
                ReturnType = MLIRTypes.i32
                Implementation = Catalog.mathMin
            })
            
            ("abs", { 
                Name = "abs"
                ParameterTypes = [MLIRTypes.i32]
                ReturnType = MLIRTypes.i32
                Implementation = Catalog.mathAbs
            })
            
            // Array operations
            ("Array.create", { 
                Name = "Array.create"
                ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
                ReturnType = MLIRTypes.memref MLIRTypes.i32
                Implementation = Catalog.arrayCreate
            })
            
            ("Array.zeroCreate", { 
                Name = "Array.zeroCreate"
                ParameterTypes = [MLIRTypes.i32]
                ReturnType = MLIRTypes.memref MLIRTypes.i32
                Implementation = arrayZeroCreate
            })
            
            ("Array.get", { 
                Name = "Array.get"
                ParameterTypes = [MLIRTypes.memref MLIRTypes.i32; MLIRTypes.i32]
                ReturnType = MLIRTypes.i32
                Implementation = Catalog.arrayGet
            })
            
            ("Array.set", { 
                Name = "Array.set"
                ParameterTypes = [MLIRTypes.memref MLIRTypes.i32; MLIRTypes.i32; MLIRTypes.i32]
                ReturnType = MLIRTypes.void_
                Implementation = Catalog.arraySet
            })
        ] 
        |> Map.ofList
    
    /// Check if a function is a built-in
    let isBuiltin (name: string): bool =
        Map.containsKey name builtinFunctions
    
    /// Get built-in function signature
    let getBuiltin (name: string): BuiltinSignature option =
        Map.tryFind name builtinFunctions
    
    /// Generate call to built-in function using XParsec validation
    let generateBuiltinCall (name: string) (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match getBuiltin name with
            | Some signature ->
                // Type check arguments
                if args.Length <> signature.ParameterTypes.Length then
                    return! fail "builtin_call" 
                        (sprintf "Function '%s' expects %d arguments, got %d" 
                         name signature.ParameterTypes.Length args.Length)
                else
                    // Validate argument types
                    let typeMatches = 
                        List.zip args signature.ParameterTypes
                        |> List.forall (fun (arg, expectedType) -> 
                            TypeAnalysis.canConvertTo arg.Type expectedType)
                    
                    if not typeMatches then
                        return! fail "builtin_call" 
                            (sprintf "Type mismatch in arguments for function '%s'" name)
                    else
                        return! signature.Implementation args
            | None ->
                return! fail "builtin_call" (sprintf "Unknown built-in function: %s" name)
        }

/// High-level operations using XParsec builders
module Operations =
    
    /// Console I/O operations with proper MLIR syntax
    let printInt (value: MLIRValue): MLIRCombinator<MLIRValue> =
        mlir {
            do! requireExternal "printf"
            let! result = nextSSA "print"
            
            // Get format string address
            let! fmtAddr = nextSSA "fmt_addr"
            do! emitLine (sprintf "%s = llvm.mlir.addressof @d_fmt : !llvm.ptr" fmtAddr)
            
            // Build printf call
            let printOp = buildFuncCall result "printf" [fmtAddr; value.SSA] "(!llvm.ptr, i32) -> i32"
            do! emitLine printOp
            
            return! Constants.unitConstant
        }
    
    let printString (value: MLIRValue): MLIRCombinator<MLIRValue> =
        mlir {
            do! requireExternal "printf"
            let! result = nextSSA "print"
            
            // Get format string address
            let! fmtAddr = nextSSA "fmt_addr"
            do! emitLine (sprintf "%s = llvm.mlir.addressof @s_fmt : !llvm.ptr" fmtAddr)
            
            // Build printf call
            let printOp = buildFuncCall result "printf" [fmtAddr; value.SSA] "(!llvm.ptr, !llvm.ptr) -> i32"
            do! emitLine printOp
            
            return! Constants.unitConstant
        }
    
    /// Memory management with XParsec-validated operations
    let stackAlloc (elementType: MLIRType) (count: MLIRValue): MLIRCombinator<MLIRValue> =
        mlir {
            let! ptr = nextSSA "stack_array"
            
            let allocaOp = buildAlloca ptr (Core.formatType elementType) (Some count.SSA)
            do! emitLine allocaOp
            do! Core.emitComment (sprintf "Stack allocated %s array" (Core.formatType elementType))
            
            return Core.createValue ptr (MLIRTypes.memref elementType)
        }

/// Format string constants using XParsec builders
module FormatStrings =
    
    /// Helper to sequence a list of monadic operations
    let rec sequenceM operations =
        mlir {
            match operations with
            | [] -> return ()
            | op :: rest ->
                do! op
                return! sequenceM rest
        }
    
    /// Emit format string globals with proper MLIR syntax
    let emitFormatStrings: MLIRCombinator<unit> =
        mlir {
            // Build global constants using XParsec combinators
            let globalDeclarations = [
                buildGlobalConstant "d_fmt" "\"%d\\00\"" "!llvm.array<4 x i8>"
                buildGlobalConstant "s_fmt" "\"%s\\00\"" "!llvm.array<4 x i8>"  
                buildGlobalConstant "ld_fmt" "\"%ld\\00\"" "!llvm.array<5 x i8>"
                buildGlobalConstant "f_fmt" "\"%f\\00\"" "!llvm.array<4 x i8>"
                buildGlobalConstant "newline_fmt" "\"\\n\\00\"" "!llvm.array<3 x i8>"
            ]
            
            // Emit all global declarations
            do! sequenceM (globalDeclarations |> List.map emitLine)
            
            // Mark these as external symbols we depend on
            let externalSymbols = ["d_fmt"; "s_fmt"; "ld_fmt"; "f_fmt"; "newline_fmt"]
            do! sequenceM (externalSymbols |> List.map requireExternal)
        }