module Alex.CodeGeneration.MLIROperatorGenerator

open Core.Types.TypeSystem
open Core.XParsec.Foundation
open Alex.CodeGeneration.MLIREmitter
open Alex.CodeGeneration.MLIRBuiltins

/// Operator classification for proper MLIR dialect mapping
type OperatorClass =
    | Arithmetic of string  // arith dialect operation
    | Comparison of string  // arith comparison operation
    | Logical of string     // logical operation
    | Bitwise of string     // bitwise operation

/// Operator signature with type information
type OperatorSignature = {
    Symbol: string
    Class: OperatorClass
    InputTypes: MLIRType list
    OutputType: MLIRType
    Generator: MLIRValue list -> MLIRBuilder<MLIRValue>
}



/// Binary operation helper module
module BinaryOps =

    
    /// Generate binary arithmetic operation
    let binaryArithOp (op: string) (left: MLIRValue) (right: MLIRValue): MLIRBuilder<MLIRValue> =
        mlir {
            let! result = nextSSA (op.Replace(".", "_"))
            let leftTypeStr = left.Type
            do! emitLine (sprintf "%s = arith.%s %s, %s : %s" result op left.SSA right.SSA leftTypeStr)
            return createValue result (parseTypeFromString left.Type)
        }
    
    /// Generate comparison operation
    let compare (predicate: string) (left: MLIRValue) (right: MLIRValue): MLIRBuilder<MLIRValue> =
        mlir {
            let! result = nextSSA "cmp"
            let leftTypeStr = left.Type
            do! emitLine (sprintf "%s = arith.cmpi %s, %s, %s : %s" result predicate left.SSA right.SSA leftTypeStr)
            return createValue result MLIRTypes.i1
        }
    
    /// Generate floating point comparison
    let compareFloat (predicate: string) (left: MLIRValue) (right: MLIRValue): MLIRBuilder<MLIRValue> =
        mlir {
            let! result = nextSSA "fcmp"
            let leftTypeStr = left.Type
            do! emitLine (sprintf "%s = arith.cmpf %s, %s, %s : %s" result predicate left.SSA right.SSA leftTypeStr)
            return createValue result MLIRTypes.i1
        }



/// Integer arithmetic operations using Foundation patterns
module Arithmetic =
    
    let intAdd (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.binaryArithOp "addi" left right
            | _ -> return! failHard "int_add" "Expected exactly 2 arguments"
        }
    
    let intSub (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.binaryArithOp "subi" left right
            | _ -> return! failHard "int_sub" "Expected exactly 2 arguments"
        }
    
    let intMul (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.binaryArithOp "muli" left right
            | _ -> return! failHard "int_mul" "Expected exactly 2 arguments"
        }
    
    let intDiv (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.binaryArithOp "divsi" left right
            | _ -> return! failHard "int_div" "Expected exactly 2 arguments"
        }
    
    let intMod (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.binaryArithOp "remsi" left right
            | _ -> return! failHard "int_mod" "Expected exactly 2 arguments"
        }
    
    /// Float arithmetic operations
    let floatAdd (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.binaryArithOp "addf" left right
            | _ -> return! failHard "float_add" "Expected exactly 2 arguments"
        }
    
    let floatSub (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.binaryArithOp "subf" left right
            | _ -> return! failHard "float_sub" "Expected exactly 2 arguments"
        }
    
    let floatMul (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.binaryArithOp "mulf" left right
            | _ -> return! failHard "float_mul" "Expected exactly 2 arguments"
        }
    
    let floatDiv (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.binaryArithOp "divf" left right
            | _ -> return! failHard "float_div" "Expected exactly 2 arguments"
        }

/// Comparison operators using Foundation patterns
module Comparison =
    
    let intEqual (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.compare "eq" left right
            | _ -> return! failHard "int_equal" "Expected exactly 2 arguments"
        }
    
    let intNotEqual (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.compare "ne" left right
            | _ -> return! failHard "int_not_equal" "Expected exactly 2 arguments"
        }
    
    let intLessThan (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.compare "slt" left right
            | _ -> return! failHard "int_less_than" "Expected exactly 2 arguments"
        }
    
    let intLessEqual (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.compare "sle" left right
            | _ -> return! failHard "int_less_equal" "Expected exactly 2 arguments"
        }
    
    let intGreaterThan (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.compare "sgt" left right
            | _ -> return! failHard "int_greater_than" "Expected exactly 2 arguments"
        }
    
    let intGreaterEqual (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.compare "sge" left right
            | _ -> return! failHard "int_greater_equal" "Expected exactly 2 arguments"
        }
    
    /// Float comparison operations
    let floatEqual (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.compareFloat "oeq" left right
            | _ -> return! failHard "float_equal" "Expected exactly 2 arguments"
        }
    
    let floatLessThan (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.compareFloat "olt" left right
            | _ -> return! failHard "float_less_than" "Expected exactly 2 arguments"
        }
    
    let floatLessEqual (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.compareFloat "ole" left right
            | _ -> return! failHard "float_less_equal" "Expected exactly 2 arguments"
        }
    
    let floatGreaterThan (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.compareFloat "ogt" left right
            | _ -> return! failHard "float_greater_than" "Expected exactly 2 arguments"
        }
    
    let floatGreaterEqual (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.compareFloat "oge" left right
            | _ -> return! failHard "float_greater_equal" "Expected exactly 2 arguments"
        }

/// Logical operators using combinators
module Logical =
    
    let logicalAnd (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] ->
                let! result = nextSSA "and"
                do! emitLine (sprintf "%s = arith.andi %s, %s : i1" result left.SSA right.SSA)
                return createValue result MLIRTypes.i1
            | _ -> return! failHard "logical_and" "Expected exactly 2 arguments"
        }
    
    let logicalOr (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] ->
                let! result = nextSSA "or"
                do! emitLine (sprintf "%s = arith.ori %s, %s : i1" result left.SSA right.SSA)
                return createValue result MLIRTypes.i1
            | _ -> return! failHard "logical_or" "Expected exactly 2 arguments"
        }
    
    let logicalNot (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [value] ->
                let! result = nextSSA "not"
                let! trueConst = Constants.intConstant 1 1
                do! emitLine (sprintf "%s = arith.xori %s, %s : i1" result value.SSA trueConst.SSA)
                return createValue result MLIRTypes.i1
            | _ -> return! failHard "logical_not" "Expected exactly 1 argument"
        }

/// Bitwise operators using Foundation patterns
module Bitwise =
    
    let bitwiseAnd (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] ->
                let! result = nextSSA "band"
                let typeStr = left.Type
                do! emitLine (sprintf "%s = arith.andi %s, %s : %s" result left.SSA right.SSA typeStr)
                return createValue result (parseTypeFromString left.Type)
            | _ -> return! failHard "bitwise_and" "Expected exactly 2 arguments"
        }
    
    let bitwiseOr (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] ->
                let! result = nextSSA "bor"
                let typeStr = left.Type
                do! emitLine (sprintf "%s = arith.ori %s, %s : %s" result left.SSA right.SSA typeStr)
                return createValue result (parseTypeFromString left.Type)
            | _ -> return! failHard "bitwise_or" "Expected exactly 2 arguments"
        }
    
    let bitwiseXor (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [left; right] ->
                let! result = nextSSA "bxor"
                let typeStr = left.Type
                do! emitLine (sprintf "%s = arith.xori %s, %s : %s" result left.SSA right.SSA typeStr)
                return createValue result (parseTypeFromString left.Type)
            | _ -> return! failHard "bitwise_xor" "Expected exactly 2 arguments"
        }
    
    let bitwiseNot (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [value] ->
                let! result = nextSSA "bnot"
                let typeStr = value.Type
                let valueType = parseTypeFromString value.Type
                
                // XOR with all ones to flip all bits
                let allOnes = match valueType.BitWidth with
                                | Some 32 -> -1
                                | Some 64 -> -1
                                | Some 16 -> 65535
                                | Some 8 -> 255
                                | Some 1 -> 1
                                | _ -> -1
                let bitWidth = valueType.BitWidth |> Option.defaultValue 32
                let! onesConst = Constants.intConstant allOnes bitWidth
                do! emitLine (sprintf "%s = arith.xori %s, %s : %s" result value.SSA onesConst.SSA typeStr)
                return createValue result valueType
            | _ -> return! failHard "bitwise_not" "Expected exactly 1 argument"
        }
    
    let shiftLeft (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [value; amount] ->
                let! result = nextSSA "shl"
                let typeStr = value.Type
                do! emitLine (sprintf "%s = arith.shli %s, %s : %s" result value.SSA amount.SSA typeStr)
                return createValue result (parseTypeFromString value.Type)
            | _ -> return! failHard "shift_left" "Expected exactly 2 arguments"
        }
    
    let shiftRight (args: MLIRValue list): MLIRBuilder<MLIRValue> =
        mlir {
            match args with
            | [value; amount] ->
                let! result = nextSSA "shr"
                let typeStr = value.Type
                do! emitLine (sprintf "%s = arith.shrsi %s, %s : %s" result value.SSA amount.SSA typeStr)
                return createValue result (parseTypeFromString value.Type)
            | _ -> return! failHard "shift_right" "Expected exactly 2 arguments"
        }

/// Operator registry using proper Foundation patterns
module Registry =
    
    /// Helper to create operator signatures
    let createOp symbol opClass inputTypes outputType generator =
        (symbol, {
            Symbol = symbol
            Class = opClass
            InputTypes = inputTypes
            OutputType = outputType
            Generator = generator
        })
    
    /// Arithmetic operators
    let arithmeticOps = [
        createOp "+" (Arithmetic "addi") [MLIRTypes.i32; MLIRTypes.i32] MLIRTypes.i32 Arithmetic.intAdd
        createOp "-" (Arithmetic "subi") [MLIRTypes.i32; MLIRTypes.i32] MLIRTypes.i32 Arithmetic.intSub
        createOp "*" (Arithmetic "muli") [MLIRTypes.i32; MLIRTypes.i32] MLIRTypes.i32 Arithmetic.intMul
        createOp "/" (Arithmetic "divsi") [MLIRTypes.i32; MLIRTypes.i32] MLIRTypes.i32 Arithmetic.intDiv
        createOp "%" (Arithmetic "remsi") [MLIRTypes.i32; MLIRTypes.i32] MLIRTypes.i32 Arithmetic.intMod
    ]
    
    /// Float arithmetic operators
    let floatArithmeticOps = [
        createOp "+." (Arithmetic "addf") [MLIRTypes.f32; MLIRTypes.f32] MLIRTypes.f32 Arithmetic.floatAdd
        createOp "-." (Arithmetic "subf") [MLIRTypes.f32; MLIRTypes.f32] MLIRTypes.f32 Arithmetic.floatSub
        createOp "*." (Arithmetic "mulf") [MLIRTypes.f32; MLIRTypes.f32] MLIRTypes.f32 Arithmetic.floatMul
        createOp "/." (Arithmetic "divf") [MLIRTypes.f32; MLIRTypes.f32] MLIRTypes.f32 Arithmetic.floatDiv
    ]
    
    /// Comparison operators
    let comparisonOps = [
        createOp "=" (Comparison "eq") [MLIRTypes.i32; MLIRTypes.i32] MLIRTypes.i1 Comparison.intEqual
        createOp "<>" (Comparison "ne") [MLIRTypes.i32; MLIRTypes.i32] MLIRTypes.i1 Comparison.intNotEqual
        createOp "<" (Comparison "slt") [MLIRTypes.i32; MLIRTypes.i32] MLIRTypes.i1 Comparison.intLessThan
        createOp "<=" (Comparison "sle") [MLIRTypes.i32; MLIRTypes.i32] MLIRTypes.i1 Comparison.intLessEqual
        createOp ">" (Comparison "sgt") [MLIRTypes.i32; MLIRTypes.i32] MLIRTypes.i1 Comparison.intGreaterThan
        createOp ">=" (Comparison "sge") [MLIRTypes.i32; MLIRTypes.i32] MLIRTypes.i1 Comparison.intGreaterEqual
    ]
    
    /// Float comparison operators
    let floatComparisonOps = [
        createOp "=." (Comparison "oeq") [MLIRTypes.f32; MLIRTypes.f32] MLIRTypes.i1 Comparison.floatEqual
        createOp "<." (Comparison "olt") [MLIRTypes.f32; MLIRTypes.f32] MLIRTypes.i1 Comparison.floatLessThan
        createOp "<=." (Comparison "ole") [MLIRTypes.f32; MLIRTypes.f32] MLIRTypes.i1 Comparison.floatLessEqual
        createOp ">." (Comparison "ogt") [MLIRTypes.f32; MLIRTypes.f32] MLIRTypes.i1 Comparison.floatGreaterThan
        createOp ">=." (Comparison "oge") [MLIRTypes.f32; MLIRTypes.f32] MLIRTypes.i1 Comparison.floatGreaterEqual
    ]
    
    /// Logical operators
    let logicalOps = [
        createOp "&&" (Logical "andi") [MLIRTypes.i1; MLIRTypes.i1] MLIRTypes.i1 Logical.logicalAnd
        createOp "||" (Logical "ori") [MLIRTypes.i1; MLIRTypes.i1] MLIRTypes.i1 Logical.logicalOr
        createOp "not" (Logical "xori") [MLIRTypes.i1] MLIRTypes.i1 Logical.logicalNot
    ]
    
    /// Bitwise operators
    let bitwiseOps = [
        createOp "&&&" (Bitwise "andi") [MLIRTypes.i32; MLIRTypes.i32] MLIRTypes.i32 Bitwise.bitwiseAnd
        createOp "|||" (Bitwise "ori") [MLIRTypes.i32; MLIRTypes.i32] MLIRTypes.i32 Bitwise.bitwiseOr
        createOp "^^^" (Bitwise "xori") [MLIRTypes.i32; MLIRTypes.i32] MLIRTypes.i32 Bitwise.bitwiseXor
        createOp "~~~" (Bitwise "xori") [MLIRTypes.i32] MLIRTypes.i32 Bitwise.bitwiseNot
        createOp "<<<" (Bitwise "shli") [MLIRTypes.i32; MLIRTypes.i32] MLIRTypes.i32 Bitwise.shiftLeft
        createOp ">>>" (Bitwise "shrsi") [MLIRTypes.i32; MLIRTypes.i32] MLIRTypes.i32 Bitwise.shiftRight
    ]
    
    /// All operators registry
    let allOperators: Map<string, OperatorSignature> =
        (arithmeticOps @ floatArithmeticOps @ comparisonOps @ floatComparisonOps @ logicalOps @ bitwiseOps)
        |> Map.ofList
    
    /// Lookup operator by symbol
    let tryFindOperator (symbol: string) : OperatorSignature option =
        Map.tryFind symbol allOperators
    
    /// Check if symbol is a known operator
    let isOperator (symbol: string) : bool =
        Map.containsKey symbol allOperators
    
    /// Get all operators of a specific class
    let getOperatorsByClass (opClass: OperatorClass) : OperatorSignature list =
        allOperators
        |> Map.toList
        |> List.map snd
        |> List.filter (fun op -> op.Class = opClass)

/// Operator generation utilities
module Generation =
    
    /// Generate call to operator function
    let generateOperatorCall (symbol: string) (args: MLIRValue list) : MLIRBuilder<MLIRValue> =
        mlir {
            match Registry.tryFindOperator symbol with
            | Some op ->
                if List.length args = List.length op.InputTypes then
                    return! op.Generator args
                else
                    let errorMsg = sprintf "Operator '%s' expects %d operands, got %d" 
                                          symbol (List.length op.InputTypes) (List.length args)
                    return! failHard "operator_call" errorMsg
            | None ->
                return! failHard "operator_call" (sprintf "Unknown operator: %s" symbol)
        }
    
    /// Generate type signatures for operators (for external declarations)
    let generateOperatorSignatures : MLIRBuilder<unit> =
        mlir {
            do! emitComment "Operator function signatures"
            
            let rec emitSignatures operators =
                mlir {
                    match operators with
                    | [] -> return ()
                    | (_, op) :: rest ->
                        let paramTypeStrs = op.InputTypes |> List.map mlirTypeToString
                        let returnTypeStr = mlirTypeToString op.OutputType
                        let signature = sprintf "(%s) -> %s" (String.concat ", " paramTypeStrs) returnTypeStr
                        do! emitComment (sprintf "Operator %s: %s" op.Symbol signature)
                        return! emitSignatures rest
                }
            
            do! emitSignatures (Map.toList Registry.allOperators)
        }
    
    /// Check operator precedence and associativity
    let getOperatorPrecedence (symbol: string) : int =
        match symbol with
        | "*" | "/" | "%" | "*." | "/." -> 10
        | "+" | "-" | "+." | "-." -> 9
        | "<<<" | ">>>" -> 8
        | "&&&" -> 7
        | "^^^" -> 6
        | "|||" -> 5
        | "=" | "<>" | "<" | "<=" | ">" | ">=" -> 4
        | "=." | "<." | "<=." | ">." | ">=." -> 4
        | "&&" -> 3
        | "||" -> 2
        | "not" | "~~~" -> 15  // Unary operators have highest precedence
        | _ -> 1  // Default low precedence
    
    /// Check if operator is left-associative
    let isLeftAssociative (symbol: string) : bool =
        match symbol with
        | "not" | "~~~" -> false  // Unary operators are right-associative
        | _ -> true  // Most binary operators are left-associative