module Dabbit.CodeGeneration.MLIRBuiltins

open Core.Types.TypeSystem
open Core.XParsec.Foundation
open Dabbit.CodeGeneration.MLIREmitter

/// Built-in function registry using MLIRBuilder patterns
type BuiltinSignature = {
    Name: string
    ParameterTypes: MLIRType list
    ReturnType: MLIRType
    Implementation: MLIRValue list -> MLIRBuilder<MLIRValue>
}

/// Placeholder for missing build functions
let buildFuncCall (result: string) (funcName: string) (args: string list) (signature: string) : string =
    sprintf "%s = func.call @%s(%s) : %s" result funcName (String.concat ", " args) signature

let buildArithOp (op: string) (result: string) (left: string) (right: string) (opInfo: string) : string =
    sprintf "%s = arith.%s %s, %s : %s" result op left right opInfo

let buildSelect (result: string) (condition: string) (trueVal: string) (falseVal: string) (typ: string) : string =
    sprintf "%s = arith.select %s, %s, %s : %s" result condition trueVal falseVal typ

let buildAlloca (result: string) (elemType: string) (count: string option) : string =
    match count with
    | Some c -> sprintf "%s = memref.alloca(%s) : memref<%s>" result c elemType
    | None -> sprintf "%s = memref.alloca() : memref<%s>" result elemType

let buildLoad (result: string) (source: string) (typ: string) : string =
    sprintf "%s = memref.load %s : %s" result source typ

let buildStore (value: string) (target: string) (typ: string) : string =
    sprintf "memref.store %s, %s : %s" value target typ

let buildGlobalConstant (name: string) (value: string) (typ: string) : string =
    sprintf "llvm.mlir.global constant @%s(%s) : %s" name value typ

/// Helper to create MLIR values using proper TypeSystem functions
let createValue (ssa: string) (typ: MLIRType) : MLIRValue = {
    SSA = ssa
    Type = mlirTypeToString typ
    IsConstant = false
}

/// Format type for MLIR output using TypeSystem
let formatType (typ: MLIRType) : string =
    mlirTypeToString typ

/// Emit comment helper
let emitComment (comment: string) : MLIRBuilder<unit> =
    emitLine (sprintf "// %s" comment)

/// Constants module for generating constant values
module Constants =
    let intConstant (value: int) (bits: int) : MLIRBuilder<MLIRValue> =
        mlir {
            let! ssa = nextSSA "const"
            let intType = MLIRTypes.int bits
            do! emitLine (sprintf "%s = arith.constant %d : %s" ssa value (mlirTypeToString intType))
            return createValue ssa intType
        }
    
    let unitConstant : MLIRBuilder<MLIRValue> =
        mlir {
            return createValue "" MLIRTypes.void_
        }

/// Built-in function catalog using MLIRBuilder combinators
module Catalog =
    
    /// String operations using MLIRBuilder combinators
    let stringConcat (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] ->
                let! result = nextSSA "concat"
                do! requireExternal "strcat"
                
                let funcCallOp = buildFuncCall result "strcat" 
                                    [left.SSA; right.SSA] 
                                    "(!llvm.ptr, !llvm.ptr) -> !llvm.ptr"
                
                do! emitLine funcCallOp
                return createValue result (MLIRTypes.memref MLIRTypes.i8)
            | _ -> 
                return! failHard "string_concat" "Expected exactly 2 arguments"
        }
    
    let stringLength (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [str] ->
                let! result = nextSSA "strlen"
                do! requireExternal "strlen"
                
                let funcCallOp = buildFuncCall result "strlen" [str.SSA] "(!llvm.ptr) -> i32"
                do! emitLine funcCallOp
                
                return createValue result MLIRTypes.i32
            | _ ->
                return! failHard "string_length" "Expected exactly 1 argument"
        }
    
    /// Numeric operations using MLIRBuilder builders
    let mathMin (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] ->
                let! cmpResult = nextSSA "cmp"
                let! minResult = nextSSA "min"
                
                let cmpOp = buildArithOp "cmpi" cmpResult left.SSA right.SSA 
                               (sprintf "slt, %s" (formatType MLIRTypes.i32))
                do! emitLine cmpOp
                
                let selectOp = buildSelect minResult cmpResult left.SSA right.SSA 
                                  (formatType MLIRTypes.i32)
                do! emitLine selectOp
                
                return createValue minResult MLIRTypes.i32
            | _ ->
                return! failHard "math_min" "Expected exactly 2 arguments"
        }
    
    let mathAbs (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [value] ->
                let! zero = Constants.intConstant 0 32
                let! cmpResult = nextSSA "cmp"
                let! negResult = nextSSA "neg" 
                let! absResult = nextSSA "abs"
                
                let cmpOp = buildArithOp "cmpi" cmpResult value.SSA zero.SSA "slt, i32"
                do! emitLine cmpOp
                
                let negOp = buildArithOp "subi" negResult zero.SSA value.SSA "i32"
                do! emitLine negOp
                
                let selectOp = buildSelect absResult cmpResult negResult value.SSA "i32"
                do! emitLine selectOp
                
                return createValue absResult MLIRTypes.i32
            | _ ->
                return! failHard "math_abs" "Expected exactly 1 argument"
        }
    
    /// Memory operations
    let memoryLoad (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [ptr] ->
                let! result = nextSSA "load"
                let loadOp = buildLoad result ptr.SSA "!llvm.ptr"
                do! emitLine loadOp
                return createValue result MLIRTypes.i32 // Simplified type
            | _ ->
                return! failHard "memory_load" "Expected exactly 1 argument"
        }
    
    let memoryStore (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [value; ptr] ->
                let storeOp = buildStore value.SSA ptr.SSA "!llvm.ptr"
                do! emitLine storeOp
                return! Constants.unitConstant
            | _ ->
                return! failHard "memory_store" "Expected exactly 2 arguments"
        }
    
    /// Type introspection (placeholder)
    let getTypeInfo (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [_] ->
                // TODO: Implement proper type analysis when TypeAnalysis module exists
                return! failHard "get_type_info" "Type analysis not yet implemented"
            | _ ->
                return! failHard "get_type_info" "Expected exactly 1 argument"
        }
    
    /// IO operations using MLIRBuilder-validated operations
    let printInt (value: MLIRValue): MLIRBuilder<MLIRValue> =
        mlir {
            do! requireExternal "printf"
            let! result = nextSSA "print"
            
            let! fmtAddr = nextSSA "fmt_addr"
            do! emitLine (sprintf "%s = llvm.mlir.addressof @d_fmt : !llvm.ptr" fmtAddr)
            
            let printOp = buildFuncCall result "printf" [fmtAddr; value.SSA] "(!llvm.ptr, i32) -> i32"
            do! emitLine printOp
            
            return! Constants.unitConstant
        }
    
    let printString (value: MLIRValue): MLIRBuilder<MLIRValue> =
        mlir {
            do! requireExternal "printf"
            let! result = nextSSA "print"
            
            let! fmtAddr = nextSSA "fmt_addr"
            do! emitLine (sprintf "%s = llvm.mlir.addressof @s_fmt : !llvm.ptr" fmtAddr)
            
            let printOp = buildFuncCall result "printf" [fmtAddr; value.SSA] "(!llvm.ptr, !llvm.ptr) -> i32"
            do! emitLine printOp
            
            return! Constants.unitConstant
        }
    
    /// Memory management with MLIRBuilder-validated operations
    let stackAlloc (elementType: MLIRType) (count: MLIRValue): MLIRBuilder<MLIRValue> =
        mlir {
            let! ptr = nextSSA "stack_array"
            
            let allocaOp = buildAlloca ptr (formatType elementType) (Some count.SSA)
            do! emitLine allocaOp
            do! emitComment (sprintf "Stack allocated %s array" (formatType elementType))
            
            return createValue ptr (MLIRTypes.memref elementType)
        }

/// Format string constants using MLIRBuilder builders
module FormatStrings =
    
    /// Emit format string globals with proper MLIR syntax
    let emitFormatStrings: MLIRBuilder<unit> =
        mlir {
            let globalDeclarations = [
                buildGlobalConstant "d_fmt" "\"%d\\00\"" "!llvm.array<4 x i8>"
                buildGlobalConstant "s_fmt" "\"%s\\00\"" "!llvm.array<4 x i8>"  
                buildGlobalConstant "ld_fmt" "\"%ld\\00\"" "!llvm.array<5 x i8>"
                buildGlobalConstant "f_fmt" "\"%f\\00\"" "!llvm.array<4 x i8>"
                buildGlobalConstant "newline_fmt" "\"\\n\\00\"" "!llvm.array<3 x i8>"
            ]
            
            // Emit all global declarations using List operations instead of for loop
            let rec emitDeclarations declarations =
                mlir {
                    match declarations with
                    | [] -> return ()
                    | decl :: rest ->
                        do! emitLine decl
                        return! emitDeclarations rest
                }
            
            do! emitDeclarations globalDeclarations
            
            let externalSymbols = ["d_fmt"; "s_fmt"; "ld_fmt"; "f_fmt"; "newline_fmt"]
            
            // Process external symbols using List operations
            let rec requireExternals symbols =
                mlir {
                    match symbols with
                    | [] -> return ()
                    | symbol :: rest ->
                        do! requireExternal symbol
                        return! requireExternals rest
                }
            
            do! requireExternals externalSymbols
        }

/// Registry for built-in functions with proper typing
module Registry =
    
    /// Core arithmetic operations
    let arithmeticBuiltins: BuiltinSignature list = [
        {
            Name = "min"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            Implementation = Catalog.mathMin
        }
        {
            Name = "abs"
            ParameterTypes = [MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            Implementation = Catalog.mathAbs
        }
    ]
    
    /// String operations
    let stringBuiltins: BuiltinSignature list = [
        {
            Name = "concat"
            ParameterTypes = [MLIRTypes.string_; MLIRTypes.string_]
            ReturnType = MLIRTypes.string_
            Implementation = Catalog.stringConcat
        }
        {
            Name = "length"
            ParameterTypes = [MLIRTypes.string_]
            ReturnType = MLIRTypes.i32
            Implementation = Catalog.stringLength
        }
    ]
    
    /// Memory operations
    let memoryBuiltins: BuiltinSignature list = [
        {
            Name = "load"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            Implementation = Catalog.memoryLoad
        }
        {
            Name = "store"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.memref MLIRTypes.i32]
            ReturnType = MLIRTypes.void_
            Implementation = Catalog.memoryStore
        }
    ]
    
    /// All built-in functions registry
    let allBuiltins: BuiltinSignature list =
        arithmeticBuiltins @ stringBuiltins @ memoryBuiltins
    
    /// Lookup built-in function by name
    let tryFindBuiltin (name: string) : BuiltinSignature option =
        allBuiltins |> List.tryFind (fun b -> b.Name = name)
    
    /// Check if function name is a built-in
    let isBuiltin (name: string) : bool =
        allBuiltins |> List.exists (fun b -> b.Name = name)

/// Built-in function generation utilities
module Generation =
    
    /// Generate call to built-in function
    let generateBuiltinCall (name: string) (args: MLIRValue list) : MLIRBuilder<MLIRValue> =
        mlir {
            match Registry.tryFindBuiltin name with
            | Some builtin ->
                if List.length args = List.length builtin.ParameterTypes then
                    return! builtin.Implementation args
                else
                    let errorMsg = sprintf "Function '%s' expects %d arguments but got %d" 
                                          name (List.length builtin.ParameterTypes) (List.length args)
                    return! failHard "builtin_call" errorMsg
            | None ->
                return! failHard "builtin_call" (sprintf "Unknown built-in function: %s" name)
        }
    
    /// Generate type signatures for built-in functions (for external declarations)
    let generateBuiltinSignatures : MLIRBuilder<unit> =
        mlir {
            do! emitComment "Built-in function signatures"
            
            let rec emitSignatures builtins =
                mlir {
                    match builtins with
                    | [] -> return ()
                    | builtin :: rest ->
                        let paramTypeStrs = builtin.ParameterTypes |> List.map mlirTypeToString
                        let returnTypeStr = mlirTypeToString builtin.ReturnType
                        let signature = sprintf "(%s) -> %s" (String.concat ", " paramTypeStrs) returnTypeStr
                        do! emitLine (sprintf "func.func private @%s%s" builtin.Name signature)
                        return! emitSignatures rest
                }
            
            do! emitSignatures Registry.allBuiltins
        }