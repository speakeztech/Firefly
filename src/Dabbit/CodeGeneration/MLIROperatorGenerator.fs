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
    
    /// Complete operator registry with all F# operators
    let operators: Map<string, OperatorSignature> = [
        // Arithmetic operators
        ("+", {
            Symbol = "+"
            Class = Arithmetic "addi"
            InputTypes = [MLIRTypes.i32; MLIRTypes.i32]
            OutputType = MLIRTypes.i32
            Generator = Arithmetic.intAdd
        })
        ("-", {
            Symbol = "-"
            Class = Arithmetic "subi"
            InputTypes = [MLIRTypes.i32; MLIRTypes.i32]
            OutputType = MLIRTypes.i32
            Generator = Arithmetic.intSub
        })
        ("*", {
            Symbol = "*"
            Class = Arithmetic "muli"
            InputTypes = [MLIRTypes.i32; MLIRTypes.i32]
            OutputType = MLIRTypes.i32
            Generator = Arithmetic.intMul
        })
        ("/", {
            Symbol = "/"
            Class = Arithmetic "divsi"
            InputTypes = [MLIRTypes.i32; MLIRTypes.i32]
            OutputType = MLIRTypes.i32
            Generator = Arithmetic.intDiv
        })
        ("%", {
            Symbol = "%"
            Class = Arithmetic "remsi"
            InputTypes = [MLIRTypes.i32; MLIRTypes.i32]
            OutputType = MLIRTypes.i32
            Generator = Arithmetic.intMod
        })
        
        // Float arithmetic
        ("+.", {
            Symbol = "+."
            Class = Arithmetic "addf"
            InputTypes = [MLIRTypes.f32; MLIRTypes.f32]
            OutputType = MLIRTypes.f32
            Generator = Arithmetic.floatAdd
        })
        ("-.", {
            Symbol = "-."
            Class = Arithmetic "subf"
            InputTypes = [MLIRTypes.f32; MLIRTypes.f32]
            OutputType = MLIRTypes.f32
            Generator = Arithmetic.floatSub
        })
        ("*.", {
            Symbol = "*."
            Class = Arithmetic "mulf"
            InputTypes = [MLIRTypes.f32; MLIRTypes.f32]
            OutputType = MLIRTypes.f32
            Generator = Arithmetic.floatMul
        })
        ("/.", {
            Symbol = "/."
            Class = Arithmetic "divf"
            InputTypes = [MLIRTypes.f32; MLIRTypes.f32]
            OutputType = MLIRTypes.f32
            Generator = Arithmetic.floatDiv
        })
        
        // Comparison operators
        ("=", {
            Symbol = "="
            Class = Comparison "eq"
            InputTypes = [MLIRTypes.i32; MLIRTypes.i32]
            OutputType = MLIRTypes.i1
            Generator = Comparison.intEqual
        })
        ("<>", {
            Symbol = "<>"
            Class = Comparison "ne"
            InputTypes = [MLIRTypes.i32; MLIRTypes.i32]
            OutputType = MLIRTypes.i1
            Generator = Comparison.intNotEqual
        })
        ("<", {
            Symbol = "<"
            Class = Comparison "slt"
            InputTypes = [MLIRTypes.i32; MLIRTypes.i32]
            OutputType = MLIRTypes.i1
            Generator = Comparison.intLessThan
        })
        ("<=", {
            Symbol = "<="
            Class = Comparison "sle"
            InputTypes = [MLIRTypes.i32; MLIRTypes.i32]
            OutputType = MLIRTypes.i1
            Generator = Comparison.intLessEqual
        })
        (">", {
            Symbol = ">"
            Class = Comparison "sgt"
            InputTypes = [MLIRTypes.i32; MLIRTypes.i32]
            OutputType = MLIRTypes.i1
            Generator = Comparison.intGreaterThan
        })
        (">=", {
            Symbol = ">="
            Class = Comparison "sge"
            InputTypes = [MLIRTypes.i32; MLIRTypes.i32]
            OutputType = MLIRTypes.i1
            Generator = Comparison.intGreaterEqual
        })
        
        // Logical operators
        ("&&", {
            Symbol = "&&"
            Class = Logical "andi"
            InputTypes = [MLIRTypes.i1; MLIRTypes.i1]
            OutputType = MLIRTypes.i1
            Generator = Logical.logicalAnd
        })
        ("||", {
            Symbol = "||"
            Class = Logical "ori"
            InputTypes = [MLIRTypes.i1; MLIRTypes.i1]
            OutputType = MLIRTypes.i1
            Generator = Logical.logicalOr
        })
        ("not", {
            Symbol = "not"
            Class = Logical "xori"
            InputTypes = [MLIRTypes.i1]
            OutputType = MLIRTypes.i1
            Generator = Logical.logicalNot
        })
        
        // Bitwise operators
        ("&&&", {
            Symbol = "&&&"
            Class = Bitwise "andi"
            InputTypes = [MLIRTypes.i32; MLIRTypes.i32]
            OutputType = MLIRTypes.i32
            Generator = Bitwise.bitwiseAnd
        })
        ("|||", {
            Symbol = "|||"
            Class = Bitwise "ori"
            InputTypes = [MLIRTypes.i32; MLIRTypes.i32]
            OutputType = MLIRTypes.i32
            Generator = Bitwise.bitwiseOr
        })
        ("^^^", {
            Symbol = "^^^"
            Class = Bitwise "xori"
            InputTypes = [MLIRTypes.i32; MLIRTypes.i32]
            OutputType = MLIRTypes.i32
            Generator = Bitwise.bitwiseXor
        })
        ("~~~", {
            Symbol = "~~~"
            Class = Bitwise "xori"
            InputTypes = [MLIRTypes.i32]
            OutputType = MLIRTypes.i32
            Generator = Bitwise.bitwiseNot
        })
        ("<<<", {
            Symbol = "<<<"
            Class = Bitwise "shli"
            InputTypes = [MLIRTypes.i32; MLIRTypes.i32]
            OutputType = MLIRTypes.i32
            Generator = Bitwise.shiftLeft
        })
        (">>>", {
            Symbol = ">>>"
            Class = Bitwise "shrsi"
            InputTypes = [MLIRTypes.i32; MLIRTypes.i32]
            OutputType = MLIRTypes.i32
            Generator = Bitwise.shiftRight
        })
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

/// Higher-level operator patterns using Foundation
module Patterns =
    
    /// Handle unary operator application
    let unaryOp (symbol: string) (operand: MLIRValue): MLIRCombinator<MLIRValue> =
        Registry.generateOperator symbol [operand]
    
    /// Handle binary operator application
    let binaryOp (symbol: string) (left: MLIRValue) (right: MLIRValue): MLIRCombinator<MLIRValue> =
        Registry.generateOperator symbol [left; right]
    
    /// Handle operator chaining (left-associative)
    let chainLeft (symbol: string) (operands: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            match operands with
            | [] -> return! fail "chain_left" "No operands provided"
            | [single] -> return single
            | head :: tail ->
                let! result = tail |> List.fold (fun acc next ->
                    mlir {
                        let! current = acc
                        return! binaryOp symbol current next
                    }) (lift head)
                return result
        }
    
    /// Handle conditional operator (ternary)
    let conditional (condition: MLIRValue) (trueVal: MLIRValue) (falseVal: MLIRValue): MLIRCombinator<MLIRValue> =
        mlir {
            let! result = nextSSA "select"
            let typeStr = Core.formatType trueVal.Type
            do! emitLine (sprintf "%s = arith.select %s, %s, %s : %s" 
                         result condition.SSA trueVal.SSA falseVal.SSA typeStr)
            return Core.createValue result trueVal.Type
        }
    
    /// Handle type conversion operators
    let typeConvert (fromType: MLIRType) (toType: MLIRType) (value: MLIRValue): MLIRCombinator<MLIRValue> =
        mlir {
            if TypeAnalysis.areEqual fromType toType then
                return value  // No conversion needed
            else
                let! result = nextSSA "convert"
                let fromTypeStr = Core.formatType fromType
                let toTypeStr = Core.formatType toType
                
                match fromType.Category, toType.Category with
                | MLIRTypeCategory.Integer, MLIRTypeCategory.Integer ->
                    // Integer width conversion
                    do! emitLine (sprintf "%s = arith.extsi %s : %s to %s" result value.SSA fromTypeStr toTypeStr)
                | MLIRTypeCategory.Integer, MLIRTypeCategory.Float ->
                    // Int to float conversion
                    do! emitLine (sprintf "%s = arith.sitofp %s : %s to %s" result value.SSA fromTypeStr toTypeStr)
                | MLIRTypeCategory.Float, MLIRTypeCategory.Integer ->
                    // Float to int conversion
                    do! emitLine (sprintf "%s = arith.fptosi %s : %s to %s" result value.SSA fromTypeStr toTypeStr)
                | MLIRTypeCategory.Float, MLIRTypeCategory.Float ->
                    // Float precision conversion
                    do! emitLine (sprintf "%s = arith.extf %s : %s to %s" result value.SSA fromTypeStr toTypeStr)
                | _ ->
                    return! fail "type_convert" 
                        (sprintf "Unsupported conversion from %s to %s" fromTypeStr toTypeStr)
                
                return Core.createValue result toType
        }