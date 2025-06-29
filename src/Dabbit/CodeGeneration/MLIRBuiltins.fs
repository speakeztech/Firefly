module Dabbit.CodeGeneration.MLIRBuiltins

open Core.Types.TypeSystem
open MLIREmitter

/// Built-in function registry using XParsec patterns
type BuiltinSignature = {
    Name: string
    ParameterTypes: MLIRType list
    ReturnType: MLIRType
    Implementation: MLIRValue list -> MLIRCombinator<MLIRValue>
}

/// Built-in function catalog
module Catalog =
    
    /// String operations using combinators
    let stringConcat (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] ->
                let! result = nextSSA "concat"
                // For now, simplified concatenation - would need proper string handling
                do! requireExternal "strcat"
                do! emitLine (sprintf "%s = func.call @strcat(%s, %s) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr" 
                             result left.SSA right.SSA)
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
                do! emitLine (sprintf "%s = func.call @strlen(%s) : (!llvm.ptr) -> i32" result str.SSA)
                return Core.createValue result MLIRTypes.i32
            | _ ->
                return! fail "string_length" "Expected exactly 1 argument"
        }
    
    let stringReplace (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [str; pattern; replacement] ->
                // Simplified - return original string for now
                do! Core.emitComment "String replace operation (simplified)"
                return str
            | _ ->
                return! fail "string_replace" "Expected exactly 3 arguments"
        }
    
    /// Numeric conversion operations
    let intToString (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [value] ->
                let! result = nextSSA "int_to_str"
                let! buffer = Memory.alloca MLIRTypes.i8 (Some (Constants.intConstant 32 32 |> runMLIRCombinator |> function | Success(v, _) -> v | _ -> failwith "Failed to create constant"))
                // Simplified conversion using sprintf
                do! requireExternal "sprintf"
                do! emitLine (sprintf "%s = func.call @sprintf(%s, %%d_fmt, %s) : (!llvm.ptr, !llvm.ptr, i32) -> i32" 
                             result buffer.SSA value.SSA)
                return Core.createValue result MLIRTypes.string_
            | _ ->
                return! fail "int_to_string" "Expected exactly 1 argument"
        }
    
    let stringToInt (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [str] ->
                let! result = nextSSA "str_to_int"
                do! requireExternal "atoi"
                do! emitLine (sprintf "%s = func.call @atoi(%s) : (!llvm.ptr) -> i32" result str.SSA)
                return Core.createValue result MLIRTypes.i32
            | _ ->
                return! fail "string_to_int" "Expected exactly 1 argument"
        }
    
    /// Array operations using combinators
    let arrayCreate (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [size; initValue] ->
                let! arrayPtr = Memory.alloca initValue.Type (Some size)
                // Initialize all elements (simplified loop)
                do! Core.emitComment "Array initialization loop would go here"
                return arrayPtr
            | _ ->
                return! fail "array_create" "Expected exactly 2 arguments (size, initial_value)"
        }
    
    let arrayLength (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [array] ->
                // For now, return a constant - would need proper array metadata
                let! result = Constants.intConstant 0 32
                do! Core.emitComment "Array length lookup (metadata required)"
                return result
            | _ ->
                return! fail "array_length" "Expected exactly 1 argument"
        }
    
    let arrayGet (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [array; index] ->
                return! Memory.load array [index]
            | _ ->
                return! fail "array_get" "Expected exactly 2 arguments (array, index)"
        }
    
    let arraySet (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [array; index; value] ->
                do! Memory.store value array [index]
                return Constants.unitConstant |> runMLIRCombinator |> function | Success(v, _) -> v | _ -> failwith "Failed to create unit"
            | _ ->
                return! fail "array_set" "Expected exactly 3 arguments (array, index, value)"
        }
    
    /// Math operations using Foundation patterns
    let mathMin (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] ->
                let! cmp = BinaryOps.compare "slt" left right
                let! result = nextSSA "min"
                do! emitLine (sprintf "%s = arith.select %s, %s, %s : %s" 
                             result cmp.SSA left.SSA right.SSA (Core.formatType left.Type))
                return Core.createValue result left.Type
            | _ ->
                return! fail "math_min" "Expected exactly 2 arguments"
        }
    
    let mathMax (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] ->
                let! cmp = BinaryOps.compare "sgt" left right
                let! result = nextSSA "max"
                do! emitLine (sprintf "%s = arith.select %s, %s, %s : %s" 
                             result cmp.SSA left.SSA right.SSA (Core.formatType left.Type))
                return Core.createValue result left.Type
            | _ ->
                return! fail "math_max" "Expected exactly 2 arguments"
        }
    
    let mathAbs (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [value] ->
                let! zero = Constants.intConstant 0 32
                let! isNeg = BinaryOps.compare "slt" value zero
                let! negValue = nextSSA "neg"
                do! emitLine (sprintf "%s = arith.subi %s, %s : %s" 
                             negValue zero.SSA value.SSA (Core.formatType value.Type))
                let negValueResult = Core.createValue negValue value.Type
                let! result = nextSSA "abs"
                do! emitLine (sprintf "%s = arith.select %s, %s, %s : %s" 
                             result isNeg.SSA negValue value.SSA (Core.formatType value.Type))
                return Core.createValue result value.Type
            | _ ->
                return! fail "math_abs" "Expected exactly 1 argument"
        }

/// Built-in function registry using XParsec patterns
module Registry =
    
    /// All built-in functions defined using combinators
    let builtinFunctions: Map<string, BuiltinSignature> = [
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
        ("replace", { 
            Name = "replace"
            ParameterTypes = [MLIRTypes.string_; MLIRTypes.string_; MLIRTypes.string_]
            ReturnType = MLIRTypes.string_
            Implementation = Catalog.stringReplace
        })
        
        // Type conversions
        ("intToString", { 
            Name = "intToString"
            ParameterTypes = [MLIRTypes.i32]
            ReturnType = MLIRTypes.string_
            Implementation = Catalog.intToString
        })
        ("toString", { 
            Name = "toString"
            ParameterTypes = [MLIRTypes.i32]
            ReturnType = MLIRTypes.string_
            Implementation = Catalog.intToString
        })
        ("stringToInt", { 
            Name = "stringToInt"
            ParameterTypes = [MLIRTypes.string_]
            ReturnType = MLIRTypes.i32
            Implementation = Catalog.stringToInt
        })
        
        // Array operations
        ("Array.create", { 
            Name = "Array.create"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]  // size, init_value
            ReturnType = MLIRTypes.memref MLIRTypes.i32
            Implementation = Catalog.arrayCreate
        })
        ("Array.zeroCreate", { 
            Name = "Array.zeroCreate"
            ParameterTypes = [MLIRTypes.i32]
            ReturnType = MLIRTypes.memref MLIRTypes.i32
            Implementation = fun args -> 
                mlir {
                    match args with
                    | [size] ->
                        let! zero = Constants.intConstant 0 32
                        return! Catalog.arrayCreate [size; zero]
                    | _ ->
                        return! fail "Array.zeroCreate" "Expected exactly 1 argument"
                }
        })
        ("Array.length", { 
            Name = "Array.length"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.i32
            Implementation = Catalog.arrayLength
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
        
        // Math operations
        ("min", { 
            Name = "min"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            Implementation = Catalog.mathMin
        })
        ("max", { 
            Name = "max"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            Implementation = Catalog.mathMax
        })
        ("abs", { 
            Name = "abs"
            ParameterTypes = [MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            Implementation = Catalog.mathAbs
        })
    ] |> Map.ofList
    
    /// Check if a function is a built-in
    let isBuiltin (name: string): bool =
        Map.containsKey name builtinFunctions
    
    /// Get built-in function signature
    let getBuiltin (name: string): BuiltinSignature option =
        Map.tryFind name builtinFunctions
    
    /// Generate call to built-in function using combinators
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
                    let typeMatches = List.zip args signature.ParameterTypes
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

/// High-level built-in operations using Foundation patterns
module Operations =
    
    /// Console I/O operations
    let printInt (value: MLIRValue): MLIRCombinator<MLIRValue> =
        mlir {
            do! requireExternal "printf"
            let! result = nextSSA "print"
            do! emitLine (sprintf "%s = func.call @printf(%%d_fmt, %s) : (!llvm.ptr, i32) -> i32" 
                         result value.SSA)
            return! Constants.unitConstant
        }
    
    let printString (value: MLIRValue): MLIRCombinator<MLIRValue> =
        mlir {
            do! requireExternal "printf"
            let! result = nextSSA "print"
            do! emitLine (sprintf "%s = func.call @printf(%%s_fmt, %s) : (!llvm.ptr, !llvm.ptr) -> i32" 
                         result value.SSA)
            return! Constants.unitConstant
        }
    
    let readLine: MLIRCombinator<MLIRValue> =
        mlir {
            let! buffer = Memory.alloca MLIRTypes.i8 (Some (Constants.intConstant 256 32 |> runMLIRCombinator |> function | Success(v, _) -> v | _ -> failwith "Failed"))
            let! stdin = nextSSA "stdin"
            do! requireExternal "fgets"
            do! emitLine (sprintf "%s = func.call @fgets(%s, 256, %%stdin) : (!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr" 
                         stdin buffer.SSA)
            return Core.createValue stdin MLIRTypes.string_
        }
    
    /// Memory management operations (stack-based)
    let stackAlloc (elementType: MLIRType) (count: MLIRValue): MLIRCombinator<MLIRValue> =
        mlir {
            let! ptr = Memory.alloca elementType (Some count)
            do! Core.emitComment (sprintf "Stack allocated %s array" (Core.formatType elementType))
            return ptr
        }
    
    /// Error handling operations
    let panic (message: MLIRValue): MLIRCombinator<MLIRValue> =
        mlir {
            do! requireExternal "printf"
            do! requireExternal "exit"
            let! printResult = nextSSA "panic_print"
            do! emitLine (sprintf "%s = func.call @printf(%%s_fmt, %s) : (!llvm.ptr, !llvm.ptr) -> i32" 
                         printResult message.SSA)
            let! exitCode = Constants.intConstant 1 32
            let! exitResult = nextSSA "panic_exit"
            do! emitLine (sprintf "%s = func.call @exit(%s) : (i32) -> void" exitResult exitCode.SSA)
            return! Constants.unitConstant
        }

/// Format string constants for common operations
module FormatStrings =
    
    /// Emit required format string globals
    let emitFormatStrings: MLIRCombinator<unit> =
        mlir {
            do! emitLine "llvm.mlir.global internal constant @d_fmt(\"%d\\00\") : !llvm.array<4 x i8>"
            do! emitLine "llvm.mlir.global internal constant @s_fmt(\"%s\\00\") : !llvm.array<4 x i8>"
            do! emitLine "llvm.mlir.global internal constant @ld_fmt(\"%ld\\00\") : !llvm.array<5 x i8>"
            do! emitLine "llvm.mlir.global internal constant @f_fmt(\"%f\\00\") : !llvm.array<4 x i8>"
            do! emitLine "llvm.mlir.global internal constant @newline_fmt(\"\\n\\00\") : !llvm.array<3 x i8>"
            
            // External references to format strings
            do! requireExternal "d_fmt"
            do! requireExternal "s_fmt"
            do! requireExternal "ld_fmt"
            do! requireExternal "f_fmt"
            do! requireExternal "newline_fmt"
        }