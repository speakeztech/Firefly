module Dabbit.CodeGeneration.MLIROperatorGenerator

open FSharp.Compiler.Syntax
open Core.Types.TypeSystem
open MLIREmitter

/// Operator classification for different MLIR dialects
type OperatorClass =
    | Arithmetic of string  // arith dialect
    | Comparison of string  // arith.cmpi
    | Logical of string     // arith dialect boolean ops
    | Bitwise of string     // arith dialect bitwise
    | Memory of string      // memref dialect
    | Control of string     // scf/cf dialects

/// Operator signature using XParsec patterns
type OperatorSignature = {
    Symbol: string
    Class: OperatorClass
    InputTypes: MLIRType list
    OutputType: MLIRType
    Generator: MLIRValue list -> MLIRCombinator<MLIRValue>
}





/// Arithmetic operators using Foundation combinators
module Arithmetic =
    
    /// Integer arithmetic operations
    let intAdd (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.add left right
            | _ -> return! fail "int_add" "Expected exactly 2 arguments"
        }
    
    let intSub (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.sub left right
            | _ -> return! fail "int_sub" "Expected exactly 2 arguments"
        }
    
    let intMul (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.mul left right
            | _ -> return! fail "int_mul" "Expected exactly 2 arguments"
        }
    
    let intDiv (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.div left right
            | _ -> return! fail "int_div" "Expected exactly 2 arguments"
        }
    
    let intMod (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] ->
                let! result = nextSSA "mod"
                let typeStr = Core.formatType left.Type
                do! emitLine (sprintf "%s = arith.remsi %s, %s : %s" result left.SSA right.SSA typeStr)
                return Core.createValue result left.Type
            | _ -> return! fail "int_mod" "Expected exactly 2 arguments"
        }
    
    /// Float arithmetic operations
    let floatAdd (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] ->
                let! result = nextSSA "fadd"
                let typeStr = Core.formatType left.Type
                do! emitLine (sprintf "%s = arith.addf %s, %s : %s" result left.SSA right.SSA typeStr)
                return Core.createValue result left.Type
            | _ -> return! fail "float_add" "Expected exactly 2 arguments"
        }
    
    let floatSub (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] ->
                let! result = nextSSA "fsub"
                let typeStr = Core.formatType left.Type
                do! emitLine (sprintf "%s = arith.subf %s, %s : %s" result left.SSA right.SSA typeStr)
                return Core.createValue result left.Type
            | _ -> return! fail "float_sub" "Expected exactly 2 arguments"
        }
    
    let floatMul (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] ->
                let! result = nextSSA "fmul"
                let typeStr = Core.formatType left.Type
                do! emitLine (sprintf "%s = arith.mulf %s, %s : %s" result left.SSA right.SSA typeStr)
                return Core.createValue result left.Type
            | _ -> return! fail "float_mul" "Expected exactly 2 arguments"
        }
    
    let floatDiv (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] ->
                let! result = nextSSA "fdiv"
                let typeStr = Core.formatType left.Type
                do! emitLine (sprintf "%s = arith.divf %s, %s : %s" result left.SSA right.SSA typeStr)
                return Core.createValue result left.Type
            | _ -> return! fail "float_div" "Expected exactly 2 arguments"
        }

/// Comparison operators using Foundation patterns
module Comparison =
    
    let intEqual (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.compare "eq" left right
            | _ -> return! fail "int_equal" "Expected exactly 2 arguments"
        }
    
    let intNotEqual (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.compare "ne" left right
            | _ -> return! fail "int_not_equal" "Expected exactly 2 arguments"
        }
    
    let intLessThan (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.compare "slt" left right
            | _ -> return! fail "int_less_than" "Expected exactly 2 arguments"
        }
    
    let intLessEqual (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.compare "sle" left right
            | _ -> return! fail "int_less_equal" "Expected exactly 2 arguments"
        }
    
    let intGreaterThan (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.compare "sgt" left right
            | _ -> return! fail "int_greater_than" "Expected exactly 2 arguments"
        }
    
    let intGreaterEqual (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] -> return! BinaryOps.compare "sge" left right
            | _ -> return! fail "int_greater_equal" "Expected exactly 2 arguments"
        }
    
    /// Float comparison operations
    let floatEqual (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] ->
                let! result = nextSSA "fcmp"
                let typeStr = Core.formatType left.Type
                do! emitLine (sprintf "%s = arith.cmpf oeq, %s, %s : %s" result left.SSA right.SSA typeStr)
                return Core.createValue result MLIRTypes.i1
            | _ -> return! fail "float_equal" "Expected exactly 2 arguments"
        }
    
    let floatLessThan (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] ->
                let! result = nextSSA "fcmp"
                let typeStr = Core.formatType left.Type
                do! emitLine (sprintf "%s = arith.cmpf olt, %s, %s : %s" result left.SSA right.SSA typeStr)
                return Core.createValue result MLIRTypes.i1
            | _ -> return! fail "float_less_than" "Expected exactly 2 arguments"
        }

/// Logical operators using combinators
module Logical =
    
    let logicalAnd (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] ->
                let! result = nextSSA "and"
                do! emitLine (sprintf "%s = arith.andi %s, %s : i1" result left.SSA right.SSA)
                return Core.createValue result MLIRTypes.i1
            | _ -> return! fail "logical_and" "Expected exactly 2 arguments"
        }
    
    let logicalOr (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] ->
                let! result = nextSSA "or"
                do! emitLine (sprintf "%s = arith.ori %s, %s : i1" result left.SSA right.SSA)
                return Core.createValue result MLIRTypes.i1
            | _ -> return! fail "logical_or" "Expected exactly 2 arguments"
        }
    
    let logicalNot (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [value] ->
                let! trueConst = Constants.boolConstant true
                let! result = nextSSA "not"
                do! emitLine (sprintf "%s = arith.xori %s, %s : i1" result value.SSA trueConst.SSA)
                return Core.createValue result MLIRTypes.i1
            | _ -> return! fail "logical_not" "Expected exactly 1 argument"
        }

/// Bitwise operators using Foundation patterns
module Bitwise =
    
    let bitwiseAnd (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] ->
                let! result = nextSSA "band"
                let typeStr = Core.formatType left.Type
                do! emitLine (sprintf "%s = arith.andi %s, %s : %s" result left.SSA right.SSA typeStr)
                return Core.createValue result left.Type
            | _ -> return! fail "bitwise_and" "Expected exactly 2 arguments"
        }
    
    let bitwiseOr (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] ->
                let! result = nextSSA "bor"
                let typeStr = Core.formatType left.Type
                do! emitLine (sprintf "%s = arith.ori %s, %s : %s" result left.SSA right.SSA typeStr)
                return Core.createValue result left.Type
            | _ -> return! fail "bitwise_or" "Expected exactly 2 arguments"
        }
    
    let bitwiseXor (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [left; right] ->
                let! result = nextSSA "bxor"
                let typeStr = Core.formatType left.Type
                do! emitLine (sprintf "%s = arith.xori %s, %s : %s" result left.SSA right.SSA typeStr)
                return Core.createValue result left.Type
            | _ -> return! fail "bitwise_xor" "Expected exactly 2 arguments"
        }
    
    let bitwiseNot (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [value] ->
                let! result = nextSSA "bnot"
                let typeStr = Core.formatType value.Type
                // XOR with all ones to flip all bits
                let allOnes = match value.Type.Width with
                                | Some 32 -> -1
                                | Some 64 -> -1L |> int
                                | Some 8 -> 255
                                | _ -> -1
                let! onesConst = Constants.intConstant allOnes (value.Type.Width |> Option.defaultValue 32)
                do! emitLine (sprintf "%s = arith.xori %s, %s : %s" result value.SSA onesConst.SSA typeStr)
                return Core.createValue result value.Type
            | _ -> return! fail "bitwise_not" "Expected exactly 1 argument"
        }
    
    let shiftLeft (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [value; amount] ->
                let! result = nextSSA "shl"
                let typeStr = Core.formatType value.Type
                do! emitLine (sprintf "%s = arith.shli %s, %s : %s" result value.SSA amount.SSA typeStr)
                return Core.createValue result value.Type
            | _ -> return! fail "shift_left" "Expected exactly 2 arguments"
        }
    
    let shiftRight (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match args with
            | [value; amount] ->
                let! result = nextSSA "shr"
                let typeStr = Core.formatType value.Type
                do! emitLine (sprintf "%s = arith.shrsi %s, %s : %s" result value.SSA amount.SSA typeStr)
                return Core.createValue result value.Type
            | _ -> return! fail "shift_right" "Expected exactly 2 arguments"
        }

/// Operator registry using XParsec patterns
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
    
    /// Complete operator registry combining all categories
    let operators: Map<string, OperatorSignature> = 
        List.concat [
            arithmeticOps
            floatArithmeticOps
            comparisonOps
            logicalOps
            bitwiseOps
        ] |> Map.ofList
    
    /// Check if an operator exists
    let hasOperator (symbol: string): bool =
        Map.containsKey symbol operators
    
    /// Get operator signature
    let getOperator (symbol: string): OperatorSignature option =
        Map.tryFind symbol operators
    
    /// Generate operator call using combinators
    let generateOperator (symbol: string) (args: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match getOperator symbol with
            | Some op ->
                // Type checking
                if args.Length <> op.InputTypes.Length then
                    return! fail "operator_call" 
                        (sprintf "Operator '%s' expects %d operands, got %d" 
                         symbol op.InputTypes.Length args.Length)
                else
                    // Generate the operation
                    return! op.Generator args
            | None ->
                return! fail "operator_call" (sprintf "Unknown operator: %s" symbol)
        }