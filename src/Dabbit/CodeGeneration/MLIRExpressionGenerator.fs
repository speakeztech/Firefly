module Dabbit.CodeGeneration.MLIRExpressionGenerator

open FSharp.Compiler.Syntax
open Core.Types.TypeSystem
open Core.XParsec.Foundation
open MLIREmitter
open MLIRBuiltins
open MLIRTypeOperations
open MLIRControlFlow

/// Utility functions
let lift value = mlir { return value }

let rec mapM (f: 'a -> MLIRBuilder<'b>) (list: 'a list): MLIRBuilder<'b list> =
    mlir {
        match list with
        | [] -> return []
        | head :: tail ->
            let! mappedHead = f head
            let! mappedTail = mapM f tail
            return mappedHead :: mappedTail
    }

let (|>>) (m: MLIRBuilder<'a>) (f: 'a -> 'b): MLIRBuilder<'b> =
    mlir {
        let! value = m
        return f value
    }

// Define helper modules that don't depend on the recursive parts
module MemRefOps =
    let store (value: MLIRValue) (memref: MLIRValue) (indices: MLIRValue list): MLIRBuilder<unit> =
        mlir {
            let indexStr = 
                if List.isEmpty indices then ""
                else "[" + (indices |> List.map (fun idx -> idx.SSA) |> String.concat ", ") + "]"
            
            do! emitLine (sprintf "memref.store %s, %s%s : %s" value.SSA memref.SSA indexStr memref.Type)
        }
        
    let load (memref: MLIRValue) (indices: MLIRValue list) : MLIRBuilder<MLIRValue> =
        mlir {
            let! result = nextSSA "load"
            let indexStr = 
                if List.isEmpty indices then ""
                else "[" + (indices |> List.map (fun idx -> idx.SSA) |> String.concat ", ") + "]"
            
            do! emitLine (sprintf "%s = memref.load %s%s : %s" result memref.SSA indexStr memref.Type)
            
            // Extract element type from memref type
            let memrefType = parseTypeFromString memref.Type
            let elementType = 
                match memrefType.Category with
                | MemRef -> 
                    match memrefType.ElementType with
                    | Some elemType -> elemType
                    | None -> MLIRTypes.i32
                | _ -> MLIRTypes.i32
            
            return createValue result elementType
        }

// Place all the mutually dependent modules within a single module rec block
module rec ExpressionGenerators =
    
    module Constants = 
        let intConstant (value: int) (bits: int) : MLIRBuilder<MLIRValue> =
            mlir {
                let! result = nextSSA "const"
                do! emitLine (sprintf "%s = arith.constant %d : i%d" result value bits)
                return createValue result (MLIRTypes.int bits)
            }
            
        let unitConstant : MLIRBuilder<MLIRValue> =
            mlir {
                let! result = nextSSA "unit"
                do! emitLine (sprintf "%s = arith.constant 0 : i1" result)
                return createValue result MLIRTypes.void_
            }
            
        let boolConstant (value: bool) : MLIRBuilder<MLIRValue> =
            mlir {
                let intValue = if value then 1 else 0
                let! result = nextSSA "bool"
                do! emitLine (sprintf "%s = arith.constant %d : i1" result intValue)
                return createValue result MLIRTypes.i1
            }
            
        let stringConstant (value: string) : MLIRBuilder<MLIRValue> =
            mlir {
                let! result = nextSSA "str"
                do! emitLine (sprintf "%s = constant \"%s\" : !llvm.ptr<i8>" result (value.Replace("\"", "\\\"")))
                return createValue result MLIRTypes.string_
            }
            
        let floatConstant (value: float) (bits: int) : MLIRBuilder<MLIRValue> =
            mlir {
                let! result = nextSSA "float"
                do! emitLine (sprintf "%s = arith.constant %f : f%d" result value bits)
                return createValue result (if bits = 32 then MLIRTypes.f32 else MLIRTypes.f64)
            }

    module Memory =
        let alloca (typ: MLIRType) (count: MLIRValue option) : MLIRBuilder<MLIRValue> =
            mlir {
                let! ptr = nextSSA "alloca"
                let typeStr = mlirTypeToString typ
                
                // Create allocation command string outside the match
                let allocCmd = 
                    match count with
                    | Some countVal -> 
                        sprintf "%s = memref.alloca(%s) : memref<%s>" ptr countVal.SSA typeStr
                    | None -> 
                        sprintf "%s = memref.alloca() : memref<%s>" ptr typeStr
                
                // Execute the command string as a single operation
                do! emitLine allocCmd
                
                return createValue ptr (MLIRTypes.memref typ)
            }

    module StructTypes =
        let createRecord (recordType: MLIRType) (values: MLIRValue list) : MLIRBuilder<MLIRValue> =
            mlir {
                let! result = nextSSA "record"
                let! recordPtr = nextSSA "record_ptr"
                
                // Allocate memory for the record
                do! emitLine (sprintf "%s = memref.alloca() : memref<%s>" recordPtr (mlirTypeToString recordType))
                
                // Store field values - avoid for loop
                let! _ = 
                    values |> List.indexed |> mapM (fun (i, value) ->
                        mlir {
                            let! index = Constants.intConstant i 32
                            do! emitLine (sprintf "memref.store %s, %s[%s] : memref<%s>" 
                                              value.SSA recordPtr index.SSA (mlirTypeToString recordType))
                            return ()
                        })
                
                // Load the complete record
                do! emitLine (sprintf "%s = memref.load %s : memref<%s>" result recordPtr (mlirTypeToString recordType))
                return createValue result recordType
            }
            
        let accessField (record: MLIRValue) (fieldIndex: int) : MLIRBuilder<MLIRValue> =
            mlir {
                let! result = nextSSA "field"
                let! recordPtr = nextSSA "record_ptr"
                let recordType = parseTypeFromString record.Type
                
                // Store record to memory for field access
                do! emitLine (sprintf "%s = memref.alloca() : memref<%s>" recordPtr (mlirTypeToString recordType))
                do! emitLine (sprintf "memref.store %s, %s : memref<%s>" record.SSA recordPtr (mlirTypeToString recordType))
                
                // Access field by index
                let! index = Constants.intConstant fieldIndex 32
                
                // Determine field type (simplified)
                let fieldType = MLIRTypes.i32 // Default, should be determined from record type
                
                do! emitLine (sprintf "%s = memref.load %s[%s] : memref<%s>" result recordPtr index.SSA (mlirTypeToString recordType))
                return createValue result fieldType
            }
            
        let updateField (record: MLIRValue) (fieldIndex: int) (newValue: MLIRValue) : MLIRBuilder<MLIRValue> =
            mlir {
                let! result = nextSSA "updated_record"
                let! recordPtr = nextSSA "record_ptr"
                let recordType = parseTypeFromString record.Type
                
                // Create a copy of the record
                do! emitLine (sprintf "%s = memref.alloca() : memref<%s>" recordPtr (mlirTypeToString recordType))
                do! emitLine (sprintf "memref.store %s, %s : memref<%s>" record.SSA recordPtr (mlirTypeToString recordType))
                
                // Update the field
                let! index = Constants.intConstant fieldIndex 32
                do! emitLine (sprintf "memref.store %s, %s[%s] : memref<%s>" 
                                  newValue.SSA recordPtr index.SSA (mlirTypeToString recordType))
                
                // Load the updated record
                do! emitLine (sprintf "%s = memref.load %s : memref<%s>" result recordPtr (mlirTypeToString recordType))
                return createValue result recordType
            }

    module Functions =
        let call (funcName: string) (args: MLIRValue list) (returnType: MLIRType) : MLIRBuilder<MLIRValue> =
            mlir {
                let! result = nextSSA "call_result"
                
                let argsStr = args |> List.map (fun arg -> sprintf "%s" arg.SSA) |> String.concat ", "
                let argsTypesStr = args |> List.map (fun arg -> arg.Type) |> String.concat ", "
                let returnTypeStr = mlirTypeToString returnType
                
                do! emitLine (sprintf "%s = func.call @%s(%s) : (%s) -> %s" 
                                  result funcName argsStr argsTypesStr returnTypeStr)
                
                return createValue result returnType
            }

    module Registry =
        let isBuiltin (name: string) : bool =
            // Placeholder - would contain actual logic
            match name with
            | "print" | "printfn" | "printf" | "sprintf" -> true
            | _ -> false
            
        let hasOperator (op: string) : bool =
            // Placeholder - would contain actual operator detection
            match op with
            | "+" | "-" | "*" | "/" | "=" | "<>" | "<" | ">" | "<=" | ">=" -> true
            | _ -> false
            
        let generateBuiltinCall (name: string) (args: MLIRValue list) : MLIRBuilder<MLIRValue> =
            mlir {
                // Simplified implementation
                return! Functions.call (sprintf "builtin_%s" name) args MLIRTypes.void_
            }
            
        let generateOperator (op: string) (operands: MLIRValue list) : MLIRBuilder<MLIRValue> =
            mlir {
                match operands with
                | [left; right] -> 
                    let! result = nextSSA "op_result"
                    let leftType = parseTypeFromString left.Type
                    let rightType = parseTypeFromString right.Type
                    
                    // Handle type compatibility outside of conditionals
                    let compatibleTypes = areTypesEqual leftType rightType
                    let canConvertRightToLeft = canConvertImplicitly rightType leftType
                    let canConvertLeftToRight = canConvertImplicitly leftType rightType
                    
                    // Convert types if needed
                    let! leftValue = 
                        if compatibleTypes || canConvertRightToLeft then
                            mlir { return left }
                        else if canConvertLeftToRight then
                            implicitConvert leftType rightType left
                        else
                            failHard "operator_types" (sprintf "Incompatible types for operator %s: %s and %s" 
                                                    op (mlirTypeToString leftType) (mlirTypeToString rightType))
                            
                    let! rightValue =
                        if compatibleTypes || canConvertLeftToRight then
                            mlir { return right }
                        else if canConvertRightToLeft then
                            implicitConvert rightType leftType right
                        else
                            failHard "operator_types" (sprintf "Incompatible types for operator %s: %s and %s" 
                                                    op (mlirTypeToString leftType) (mlirTypeToString rightType))
                    
                    // Determine result type
                    let resultType = 
                        if compatibleTypes then leftType
                        else if canConvertRightToLeft then leftType
                        else rightType
                    
                    // Generate appropriate operation based on types and operator
                    let opCmd = 
                        match op, resultType.Category with
                        | "+", Integer ->
                            sprintf "%s = arith.addi %s, %s : %s" 
                                result leftValue.SSA rightValue.SSA (mlirTypeToString resultType)
                        | "+", Float ->
                            sprintf "%s = arith.addf %s, %s : %s" 
                                result leftValue.SSA rightValue.SSA (mlirTypeToString resultType)
                        | "-", Integer ->
                            sprintf "%s = arith.subi %s, %s : %s" 
                                result leftValue.SSA rightValue.SSA (mlirTypeToString resultType)
                        | "-", Float ->
                            sprintf "%s = arith.subf %s, %s : %s" 
                                result leftValue.SSA rightValue.SSA (mlirTypeToString resultType)
                        | "*", Integer ->
                            sprintf "%s = arith.muli %s, %s : %s" 
                                result leftValue.SSA rightValue.SSA (mlirTypeToString resultType)
                        | "*", Float ->
                            sprintf "%s = arith.mulf %s, %s : %s" 
                                result leftValue.SSA rightValue.SSA (mlirTypeToString resultType)
                        | "/", Integer ->
                            sprintf "%s = arith.divsi %s, %s : %s" 
                                result leftValue.SSA rightValue.SSA (mlirTypeToString resultType)
                        | "/", Float ->
                            sprintf "%s = arith.divf %s, %s : %s" 
                                result leftValue.SSA rightValue.SSA (mlirTypeToString resultType)
                        | "=", Integer ->
                            sprintf "%s = arith.cmpi eq, %s, %s : %s" 
                                result leftValue.SSA rightValue.SSA (mlirTypeToString resultType)
                        | "=", Float ->
                            sprintf "%s = arith.cmpf oeq, %s, %s : %s" 
                                result leftValue.SSA rightValue.SSA (mlirTypeToString resultType)
                        | "<>", Integer ->
                            sprintf "%s = arith.cmpi ne, %s, %s : %s" 
                                result leftValue.SSA rightValue.SSA (mlirTypeToString resultType)
                        | "<>", Float ->
                            sprintf "%s = arith.cmpf one, %s, %s : %s" 
                                result leftValue.SSA rightValue.SSA (mlirTypeToString resultType)
                        | "<", Integer ->
                            sprintf "%s = arith.cmpi slt, %s, %s : %s" 
                                result leftValue.SSA rightValue.SSA (mlirTypeToString resultType)
                        | "<", Float ->
                            sprintf "%s = arith.cmpf olt, %s, %s : %s" 
                                result leftValue.SSA rightValue.SSA (mlirTypeToString resultType)
                        | ">", Integer ->
                            sprintf "%s = arith.cmpi sgt, %s, %s : %s" 
                                result leftValue.SSA rightValue.SSA (mlirTypeToString resultType)
                        | ">", Float ->
                            sprintf "%s = arith.cmpf ogt, %s, %s : %s" 
                                result leftValue.SSA rightValue.SSA (mlirTypeToString resultType)
                        | "<=", Integer ->
                            sprintf "%s = arith.cmpi sle, %s, %s : %s" 
                                result leftValue.SSA rightValue.SSA (mlirTypeToString resultType)
                        | "<=", Float ->
                            sprintf "%s = arith.cmpf ole, %s, %s : %s" 
                                result leftValue.SSA rightValue.SSA (mlirTypeToString resultType)
                        | ">=", Integer ->
                            sprintf "%s = arith.cmpi sge, %s, %s : %s" 
                                result leftValue.SSA rightValue.SSA (mlirTypeToString resultType)
                        | ">=", Float ->
                            sprintf "%s = arith.cmpf oge, %s, %s : %s" 
                                result leftValue.SSA rightValue.SSA (mlirTypeToString resultType)
                        | "%", Integer ->
                            sprintf "%s = arith.remsi %s, %s : %s" 
                                result leftValue.SSA rightValue.SSA (mlirTypeToString resultType)
                        | "%", Float ->
                            sprintf "%s = arith.remf %s, %s : %s" 
                                result leftValue.SSA rightValue.SSA (mlirTypeToString resultType)
                        | _ ->
                            failwithf "Unsupported operator: %s for type %A" op resultType.Category
                    
                    // Emit the operation
                    do! emitLine opCmd
                    
                    // For comparison operations, result is i1 type
                    let finalType = 
                        match op with
                        | "=" | "<>" | "<" | ">" | "<=" | ">=" -> MLIRTypes.i1
                        | _ -> resultType
                        
                    return createValue result finalType
                    
                | [operand] ->
                    // Unary operators
                    let! result = nextSSA "unary_op"
                    let opType = parseTypeFromString operand.Type
                    
                    // Generate command based on operator and type
                    let opCmd =
                        match op, opType.Category with
                        | "-", Integer ->
                            let negResult = sprintf "%s_neg" result
                            sprintf "%s = arith.constant 0 : %s\n%s = arith.subi %s, %s : %s" 
                                result (mlirTypeToString opType) negResult result operand.SSA (mlirTypeToString opType)
                        | "-", Float ->
                            sprintf "%s = arith.negf %s : %s" 
                                result operand.SSA (mlirTypeToString opType)
                        | "not", _ ->
                            sprintf "%s = arith.xori %s, true : i1" result operand.SSA
                        | _ ->
                            failwithf "Unsupported unary operator: %s" op
                    
                    // Emit the operation
                    do! emitLine opCmd
                    
                    // Determine result type
                    let finalType =
                        match op with
                        | "not" -> MLIRTypes.i1
                        | _ -> opType
                        
                    return createValue result finalType
                        
                | _ ->
                    return! failHard "operator_arguments" (sprintf "Unsupported number of arguments for operator %s: %d" 
                                                        op operands.Length)
            }   
        
    module Mapping =
        let synTypeToMLIR (synType: SynType) : MLIRBuilder<MLIRType> =
            // Forward to the implementation in MLIRTypeOperations
            synTypeToMLIR synType

    module Core =
        /// Generate MLIR for any F# expression
        let rec generateExpression (expr: SynExpr): MLIRBuilder<MLIRValue> =
            mlir {
                match expr with
                | SynExpr.Const(constant, _) ->
                    return! generateConstant constant
                    
                | SynExpr.Ident(ident) ->
                    return! generateIdentifier ident
                    
                | SynExpr.LongIdent(_, SynLongIdent(ids, _, _), _, _) ->
                    let qualifiedName = ids |> List.map (fun id -> id.idText) |> String.concat "."
                    return! generateQualifiedIdentifier qualifiedName
                    
                | SynExpr.App(_, _, funcExpr, argExpr, _) ->
                    return! Applications.generateApplication funcExpr argExpr
                    
                | SynExpr.Paren(innerExpr, _, _, _) ->
                    return! generateExpression innerExpr
                    
                | SynExpr.Tuple(_, components, _, _) ->
                    return! Construction.generateTuple components
                    
                | SynExpr.ArrayOrList(_, exprs, _) ->
                    return! Construction.generateArray exprs
                    
                | SynExpr.Record(_, _, fields, _) ->
                    let recordFields = 
                        fields |> List.map (fun (SynExprRecordField(longId, _, expr, _)) -> 
                            SynExprRecordField(longId, None, expr, None))
                    return! Construction.generateRecord recordFields
                    
                | SynExpr.AnonRecd(_, _, fields, _, _) ->
                    let simplifiedFields = fields |> List.map (fun (id, _, expr) -> 
                        match id with
                        | SynLongIdent([ident], _, _) -> (ident, expr)
                        | _ -> failwith "Unexpected identifier format in anonymous record")
                    return! Construction.generateAnonymousRecord simplifiedFields
                    
                | SynExpr.Sequential(_, _, expr1, expr2, _, _) ->
                    return! Sequences.sequential (generateExpression expr1) (generateExpression expr2)
                    
                | SynExpr.IfThenElse(condExpr, thenExpr, elseExprOpt, _, _, _, _) ->
                    return! ControlFlow.generateIfThenElse condExpr thenExpr elseExprOpt
                    
                | SynExpr.Match(_, expr, clauses, _, _) ->
                    return! ControlFlow.generateMatch expr clauses
                    
                | SynExpr.For(_, _, ident, rangeOption, actualStartExpr, isUpTo, actualEndExpr, bodyExpr, _) ->
                    return! ControlFlow.generateFor ident actualStartExpr actualEndExpr bodyExpr
                    
                | SynExpr.ForEach(_, _, _, _, _, expr, bodyExpr, _) ->
                    return! ControlFlow.generateForEach expr bodyExpr
                    
                | SynExpr.While(_, whileExpr, doExpr, _) ->
                    return! ControlFlow.generateWhile whileExpr doExpr
                    
                | SynExpr.TryWith(tryExpr, clauses, _, _, _, _) ->
                    return! ExceptionHandling.generateTryWith tryExpr clauses
                    
                | SynExpr.TryFinally(tryExpr, finallyExpr, _, _, _, _) ->
                    return! ExceptionHandling.generateTryFinally tryExpr finallyExpr
                    
                | SynExpr.Lambda(_, _, args, bodyExpr, _, _, _) ->
                    return! Lambdas.generateLambda args bodyExpr
                    
                | SynExpr.LetOrUse(_, _, bindings, bodyExpr, _, _) ->
                    return! LetBindings.generateLet bindings bodyExpr
                    
                | SynExpr.DotGet(expr, _, SynLongIdent(ids, _, _), _) ->
                    return! FieldAccess.generateFieldAccess expr ids
                    
                | SynExpr.DotSet(expr, SynLongIdent(ids, _, _), valueExpr, _) ->
                    return! FieldAccess.generateFieldSet expr ids valueExpr
                    
                | SynExpr.New(_, typeName, argExpr, _) ->
                    return! ObjectConstruction.generateNew typeName argExpr
                    
                | _ ->
                    return! failHard "expression_generation" (sprintf "Unsupported expression type: %A" expr)
            }
        
        /// Generate constant expressions
        and generateConstant (constant: SynConst): MLIRBuilder<MLIRValue> =
            mlir {
                match constant with
                | SynConst.Int32 n -> return! Constants.intConstant n 32
                | SynConst.Int64 n -> return! Constants.intConstant (int n) 64
                | SynConst.Int16 n -> return! Constants.intConstant (int n) 16
                | SynConst.SByte n -> return! Constants.intConstant (int n) 8
                | SynConst.Byte n -> return! Constants.intConstant (int n) 8
                | SynConst.UInt32 n -> return! Constants.intConstant (int n) 32
                | SynConst.UInt64 n -> return! Constants.intConstant (int n) 64
                | SynConst.UInt16 n -> return! Constants.intConstant (int n) 16
                | SynConst.Single f -> return! Constants.floatConstant (float f) 32
                | SynConst.Double f -> return! Constants.floatConstant f 64
                | SynConst.Bool b -> return! Constants.boolConstant b
                | SynConst.String(s, _, _) -> return! Constants.stringConstant s
                | SynConst.Char c -> return! Constants.intConstant (int c) 8
                | SynConst.Unit -> return! Constants.unitConstant
                | _ -> return! failHard "constant_generation" (sprintf "Unsupported constant: %A" constant)
            }
        
        /// Generate identifier lookup
        and generateIdentifier (ident: Ident): MLIRBuilder<MLIRValue> =
            mlir {
                // Check if it's a built-in function first
                if Registry.isBuiltin ident.idText then
                    // Return function reference - simplified for now
                    let! funcRef = nextSSA (sprintf "%s_ref" ident.idText)
                    return createValue funcRef (MLIRTypes.func [] MLIRTypes.void_)
                else
                    // Check local variables
                    let! maybeLocal = lookupLocal ident.idText
                    match maybeLocal with
                    | Some (ssa, typeStr) ->
                        return createValue ssa typeStr
                    | None ->
                        return! failHard "identifier_lookup" (sprintf "Unbound identifier '%s'" ident.idText)
            }
        
        /// Generate qualified identifier lookup
        and generateQualifiedIdentifier (qualifiedName: string): MLIRBuilder<MLIRValue> =
            mlir {
                // Check if it's a built-in function
                if Registry.isBuiltin qualifiedName then
                    let! funcRef = nextSSA (sprintf "%s_ref" (qualifiedName.Replace(".", "_")))
                    return createValue funcRef (MLIRTypes.func [] MLIRTypes.void_)
                else
                    return! failHard "qualified_identifier" (sprintf "Unknown qualified identifier: %s" qualifiedName)
            }

    module Applications =
        /// Generate function application
        let rec generateApplication (funcExpr: SynExpr) (argExpr: SynExpr): MLIRBuilder<MLIRValue> =
            mlir {
                match funcExpr with
                | SynExpr.Ident(ident) when Registry.isBuiltin ident.idText ->
                    // Built-in function call
                    let! arg = Core.generateExpression argExpr
                    return! Registry.generateBuiltinCall ident.idText [arg]
                    
                | SynExpr.LongIdent(_, SynLongIdent(ids, _, _), _, _) ->
                    let qualifiedName = ids |> List.map (fun id -> id.idText) |> String.concat "."
                    if Registry.isBuiltin qualifiedName then
                        let! arg = Core.generateExpression argExpr
                        return! Registry.generateBuiltinCall qualifiedName [arg]
                    else
                        return! generateFunctionCall qualifiedName argExpr
                        
                | SynExpr.App(_, _, innerFunc, innerArg, _) ->
                    // Curried function application - collect all arguments
                    let! args = collectApplicationArgs funcExpr [argExpr]
                    return! generateCurriedCall innerFunc args
                    
                | _ ->
                    // General function call
                    let! func = Core.generateExpression funcExpr
                    let! arg = Core.generateExpression argExpr
                    return! Functions.call func.SSA [arg] MLIRTypes.void_  // TODO: Determine return type
            }
        
        /// Collect arguments from curried application
        and collectApplicationArgs (expr: SynExpr) (accArgs: SynExpr list): MLIRBuilder<SynExpr list> =
            mlir {
                match expr with
                | SynExpr.App(_, _, funcExpr, argExpr, _) ->
                    return! collectApplicationArgs funcExpr (argExpr :: accArgs)
                | _ ->
                    return accArgs
            }
        
        /// Generate curried function call
        and generateCurriedCall (funcExpr: SynExpr) (argExprs: SynExpr list): MLIRBuilder<MLIRValue> =
            mlir {
                let! func = Core.generateExpression funcExpr
                let! args = mapM Core.generateExpression argExprs
                
                match funcExpr with
                | SynExpr.Ident(ident) ->
                    return! Functions.call ident.idText args MLIRTypes.void_  // TODO: Type inference
                | _ ->
                    return! Functions.call func.SSA args MLIRTypes.void_
            }
        
        /// Generate regular function call
        and generateFunctionCall (funcName: string) (argExpr: SynExpr): MLIRBuilder<MLIRValue> =
            mlir {
                let! arg = Core.generateExpression argExpr
                return! Functions.call funcName [arg] MLIRTypes.void_  // TODO: Look up actual return type
            }

    module Operations =
        /// Detect and generate binary operations from application syntax
        let tryGenerateBinaryOp (funcExpr: SynExpr) (leftExpr: SynExpr) (rightExpr: SynExpr): MLIRBuilder<MLIRValue option> =
            mlir {
                match funcExpr with
                | SynExpr.Ident(ident) when Registry.hasOperator ident.idText ->
                    let! left = Core.generateExpression leftExpr
                    let! right = Core.generateExpression rightExpr
                    let! result = Registry.generateOperator ident.idText [left; right]
                    return Some result
                    
                | SynExpr.LongIdent(_, SynLongIdent([id], _, _), _, _) when Registry.hasOperator id.idText ->
                    let! left = Core.generateExpression leftExpr
                    let! right = Core.generateExpression rightExpr
                    let! result = Registry.generateOperator id.idText [left; right]
                    return Some result
                    
                | _ ->
                    return None
            }
        
        /// Generate unary operation
        let generateUnaryOp (operator: string) (operand: SynExpr): MLIRBuilder<MLIRValue> =
            mlir {
                let! operandValue = Core.generateExpression operand
                return! Registry.generateOperator operator [operandValue]
            }

    module Construction =
        /// Generate tuple construction
        let generateTuple (exprs: SynExpr list): MLIRBuilder<MLIRValue> =
            mlir {
                let! values = mapM Core.generateExpression exprs
                let types = values |> List.map (fun v -> parseTypeFromString v.Type)
                let tupleType = MLIRTypes.struct_ (types |> List.mapi (fun i t -> sprintf "field%d" i, t))
                
                return! StructTypes.createRecord tupleType values
            }
        
        /// Generate array construction
        let generateArray (exprs: SynExpr list): MLIRBuilder<MLIRValue> =
            mlir {
                let! values = mapM Core.generateExpression exprs
                
                match values with
                | [] ->
                    // Empty array
                    let! size = Constants.intConstant 0 32
                    return! Memory.alloca MLIRTypes.i32 (Some size)
                    
                | firstValue :: _ ->
                    let elementType = parseTypeFromString firstValue.Type
                    let arraySize = values.Length
                    let! size = Constants.intConstant arraySize 32
                    let! arrayPtr = Memory.alloca elementType (Some size)
                    
                    // Initialize array elements
                    let! _ = 
                        values |> List.indexed |> mapM (fun (i, value) ->
                            mlir {
                                let! index = Constants.intConstant i 32
                                do! MemRefOps.store value arrayPtr [index]
                                return ()
                            })
                    
                    return arrayPtr
            }
        
        /// Generate record construction
        let generateRecord (fields: SynExprRecordField list): MLIRBuilder<MLIRValue> =
            mlir {
                let extractField (field: SynExprRecordField) =
                    match field with
                    | SynExprRecordField(fieldName, _, exprOpt, _) ->
                        match exprOpt with
                        | Some expr -> Core.generateExpression expr
                        | None -> failHard "record_field" "Missing field value"
                        
                let! fieldValues = mapM extractField fields
                
                let fieldTypes = fieldValues |> List.map (fun v -> parseTypeFromString v.Type)
                let recordType = MLIRTypes.struct_ (fieldTypes |> List.mapi (fun i t -> sprintf "field%d" i, t))
                
                return! StructTypes.createRecord recordType fieldValues
            }
        
        /// Generate anonymous record construction
        let generateAnonymousRecord (fields: (Ident * SynExpr) list): MLIRBuilder<MLIRValue> =
            mlir {
                let! fieldValues = mapM (fun (_, expr) -> Core.generateExpression expr) fields
                
                let fieldTypes = fieldValues |> List.map (fun v -> parseTypeFromString v.Type)
                let recordType = MLIRTypes.struct_ (fieldTypes |> List.mapi (fun i t -> sprintf "field%d" i, t))
                
                return! StructTypes.createRecord recordType fieldValues
            }

    module ControlFlow =
        /// Generate if-then-else expression
        let generateIfThenElse (condExpr: SynExpr) (thenExpr: SynExpr) (elseExprOpt: SynExpr option): MLIRBuilder<MLIRValue> =
            mlir {
                let! condition = Core.generateExpression condExpr
                let thenBody = Core.generateExpression thenExpr
                
                let elseBody = 
                    match elseExprOpt with
                    | Some expr -> 
                        mlir {
                            let! result = Core.generateExpression expr
                            return Some result
                        }
                    | None -> mlir { return None }
                
                return! Conditionals.ifThenElse condition thenBody elseBody
            }
        
        /// Generate pattern match expression
        let generateMatch (expr: SynExpr) (clauses: SynMatchClause list): MLIRBuilder<MLIRValue> =
            mlir {
                let! scrutinee = Core.generateExpression expr
                
                let extractClause (clause: SynMatchClause) =
                    let (SynMatchClause(pat, whenExpr, resultExpr, _, _, _)) = clause
                    // TODO: Handle guard expressions (whenExpr)
                    (pat, Core.generateExpression resultExpr)
                    
                let cases = clauses |> List.map extractClause
                
                return! Patterns.matchExpression scrutinee cases
            }
        
        /// Generate for loop
        let generateFor (ident: Ident) (startExpr: SynExpr) (endExpr: SynExpr) (bodyExpr: SynExpr): MLIRBuilder<MLIRValue> =
            mlir {
                let! start = Core.generateExpression startExpr
                let! end' = Core.generateExpression endExpr
                let! step = Constants.intConstant 1 32
                
                return! Loops.forLoop ident.idText start end' step (fun loopVar ->
                    Core.generateExpression bodyExpr
                )
            }
        
        /// Generate for-each loop
        let generateForEach (collectionExpr: SynExpr) (bodyExpr: SynExpr): MLIRBuilder<MLIRValue> =
            mlir {
                let! collection = Core.generateExpression collectionExpr
                
                return! Loops.forEach "item" collection (fun itemVar ->
                    Core.generateExpression bodyExpr
                )
            }
        
        /// Generate while loop
        let generateWhile (condExpr: SynExpr) (bodyExpr: SynExpr): MLIRBuilder<MLIRValue> =
            mlir {
                let condition = Core.generateExpression condExpr
                let body = Core.generateExpression bodyExpr
                
                return! Loops.whileLoop condition body
            }

    module ExceptionHandling =
        /// Generate try-with expression
        let generateTryWith (tryExpr: SynExpr) (withCases: SynMatchClause list): MLIRBuilder<MLIRValue> =
            mlir {
                let tryBody = Core.generateExpression tryExpr
                
                let extractHandler (clause: SynMatchClause) =
                    let (SynMatchClause(pat, _, resultExpr, _, _, _)) = clause
                    (pat, Core.generateExpression resultExpr)
                    
                let handlers = withCases |> List.map extractHandler
                
                return! Exceptions.tryWith tryBody handlers
            }
        
        /// Generate try-finally expression
        let generateTryFinally (tryExpr: SynExpr) (finallyExpr: SynExpr): MLIRBuilder<MLIRValue> =
            mlir {
                let tryBody = Core.generateExpression tryExpr
                let finallyBody = mlir {
                    let! _ = Core.generateExpression finallyExpr
                    return ()
                }
                
                return! Exceptions.tryFinally tryBody finallyBody
            }

    module Lambdas =
        /// Generate lambda expression
        let generateLambda (args: SynSimplePats) (bodyExpr: SynExpr): MLIRBuilder<MLIRValue> =
            mlir {
                // For now, lambdas are converted to static functions
                // Full implementation would handle closure capture
                
                let! lambdaName = nextSSA "lambda"
                do! emitComment "Lambda expression (simplified as function reference)"
                
                // TODO: Extract argument patterns and types
                // TODO: Generate function with captured variables as parameters
                
                return createValue lambdaName (MLIRTypes.func [MLIRTypes.i32] MLIRTypes.i32)
            }

    module LetBindings =
        /// Generate let binding
        let rec generateLet (bindings: SynBinding list) (bodyExpr: SynExpr): MLIRBuilder<MLIRValue> =
            mlir {
                // Process each binding recursively to avoid for loop
                let! _ = processBindings bindings
                
                // Generate body with bindings in scope
                return! Core.generateExpression bodyExpr
            }
            
        and processBindings (bindings: SynBinding list): MLIRBuilder<unit> =
            mlir {
                match bindings with
                | [] -> return ()
                | binding :: rest ->
                    let (SynBinding(_, _, _, _, _, _, _, pat, _, expr, _, _, _)) = binding
                    let! value = Core.generateExpression expr
                    do! Bindings.bindPattern pat value
                    let! _ = processBindings rest
                    return ()
            }

    module FieldAccess =
        /// Generate field access
        let generateFieldAccess (expr: SynExpr) (fieldIds: Ident list): MLIRBuilder<MLIRValue> =
            mlir {
                let! record = Core.generateExpression expr
                let fieldName = fieldIds |> List.map (fun id -> id.idText) |> String.concat "."
                
                // For now, assume field index based on name hash
                let fieldIndex = fieldName.GetHashCode() % 8
                
                return! StructTypes.accessField record fieldIndex
            }
        
        /// Generate field assignment
        let generateFieldSet (expr: SynExpr) (fieldIds: Ident list) (valueExpr: SynExpr): MLIRBuilder<MLIRValue> =
            mlir {
                let! record = Core.generateExpression expr
                let! newValue = Core.generateExpression valueExpr
                let fieldName = fieldIds |> List.map (fun id -> id.idText) |> String.concat "."
                
                let fieldIndex = fieldName.GetHashCode() % 8
                
                return! StructTypes.updateField record fieldIndex newValue
            }

    module ObjectConstruction =
        /// Generate object construction
        let generateNew (typeName: SynType) (argExpr: SynExpr): MLIRBuilder<MLIRValue> =
            mlir {
                let! mlirType = Mapping.synTypeToMLIR typeName
                let! arg = Core.generateExpression argExpr
                
                // For now, simple allocation
                return! Memory.alloca mlirType None
            }

    module Sequences =
        /// Generate sequence of expressions
        let sequential (first: MLIRBuilder<MLIRValue>) (second: MLIRBuilder<MLIRValue>): MLIRBuilder<MLIRValue> =
            mlir {
                let! _ = first  // Execute first for side effects
                return! second  // Return result of second
            }

// Public API to generate expressions
let generateExpression (expr: SynExpr): MLIRBuilder<MLIRValue> =
    ExpressionGenerators.Core.generateExpression expr