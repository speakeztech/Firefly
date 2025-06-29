module Dabbit.CodeGeneration.MLIRExpressionGenerator

open FSharp.Compiler.Syntax
open Core.Types.TypeSystem
open Core.AST.Traversal
open MLIREmitter
open MLIRBuiltins
open MLIROperatorGenerator
open MLIRTypeOperations
open MLIRControlFlow

/// Utility functions
let lift value = mlir { return value }

let rec mapM (f: 'a -> MLIRCombinator<'b>) (list: 'a list): MLIRCombinator<'b list> =
    mlir {
        match list with
        | [] -> return []
        | head :: tail ->
            let! mappedHead = f head
            let! mappedTail = mapM f tail
            return mappedHead :: mappedTail
    }

let (|>>) (m: MLIRCombinator<'a>) (f: 'a -> 'b): MLIRCombinator<'b> =
    mlir {
        let! value = m
        return f value
    }

/// Core expression generation functions
module rec Core =
    
    /// Generate MLIR for any F# expression
    let generateExpression (expr: SynExpr): MLIRCombinator<MLIRValue> =
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
                
            | SynExpr.Tuple(_, exprs, _, _) ->
                return! Construction.generateTuple exprs
                
            | SynExpr.ArrayOrList(_, exprs, _) ->
                return! Construction.generateArray exprs
                
            | SynExpr.Record(_, _, fields, _) ->
                return! Construction.generateRecord fields
                
            | SynExpr.AnonRecd(_, _, fields, _) ->
                return! Construction.generateAnonymousRecord fields
                
            | SynExpr.Sequential(_, _, expr1, expr2, _) ->
                return! Sequences.sequential (generateExpression expr1) (generateExpression expr2)
                
            | SynExpr.IfThenElse(condExpr, thenExpr, elseExprOpt, _, _, _, _) ->
                return! ControlFlow.generateIfThenElse condExpr thenExpr elseExprOpt
                
            | SynExpr.Match(_, expr, clauses, _) ->
                return! ControlFlow.generateMatch expr clauses
                
            | SynExpr.For(_, _, ident, startExpr, _, endExpr, bodyExpr, _) ->
                return! ControlFlow.generateFor ident startExpr endExpr bodyExpr
                
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
                return! fail "expression_generation" (sprintf "Unsupported expression type: %A" expr)
        }
    
    /// Generate constant expressions
    let generateConstant (constant: SynConst): MLIRCombinator<MLIRValue> =
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
            | _ -> return! fail "constant_generation" (sprintf "Unsupported constant: %A" constant)
        }
    
    /// Generate identifier lookup
    let generateIdentifier (ident: Ident): MLIRCombinator<MLIRValue> =
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
                    // Parse type string back to MLIRType (simplified)
                    let mlirType = parseTypeString typeStr
                    return createValue ssa mlirType
                | None ->
                    return! fail "identifier_lookup" (sprintf "Unbound identifier '%s'" ident.idText)
        }
    
    /// Generate qualified identifier lookup
    let generateQualifiedIdentifier (qualifiedName: string): MLIRCombinator<MLIRValue> =
        mlir {
            // Check if it's a built-in function
            if Registry.isBuiltin qualifiedName then
                let! funcRef = nextSSA (sprintf "%s_ref" (qualifiedName.Replace(".", "_")))
                return createValue funcRef (MLIRTypes.func [] MLIRTypes.void_)
            else
                return! fail "qualified_identifier" (sprintf "Unknown qualified identifier: %s" qualifiedName)
        }
    
    /// Helper to parse type strings back to MLIRType
    let parseTypeString (typeStr: string): MLIRType =
        match typeStr with
        | "i32" -> MLIRTypes.i32
        | "i64" -> MLIRTypes.i64
        | "i16" -> MLIRTypes.i16
        | "i8" -> MLIRTypes.i8
        | "i1" -> MLIRTypes.i1
        | "f32" -> MLIRTypes.f32
        | "f64" -> MLIRTypes.f64
        | "void" -> MLIRTypes.void_
        | _ when typeStr.StartsWith("memref") -> MLIRTypes.string_  // Simplified
        | _ -> MLIRTypes.i32  // Default fallback
    
    /// Helper functions from Core module
    let createValue ssa mlirType = { SSA = ssa; Type = mlirType }
    let ssaOf (value: MLIRValue) = value.SSA
    let typeOf (value: MLIRValue) = value.Type

/// Function application and calls
module rec Applications =
    
    /// Generate function application
    let generateApplication (funcExpr: SynExpr) (argExpr: SynExpr): MLIRCombinator<MLIRValue> =
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
                return! Functions.call (Core.ssaOf func) [arg] MLIRTypes.void_  // TODO: Determine return type
        }
    
    /// Collect arguments from curried application
    let collectApplicationArgs (expr: SynExpr) (accArgs: SynExpr list): MLIRCombinator<SynExpr list> =
        mlir {
            match expr with
            | SynExpr.App(_, _, funcExpr, argExpr, _) ->
                return! collectApplicationArgs funcExpr (argExpr :: accArgs)
            | _ ->
                return accArgs
        }
    
    /// Generate curried function call
    let generateCurriedCall (funcExpr: SynExpr) (argExprs: SynExpr list): MLIRCombinator<MLIRValue> =
        mlir {
            let! func = Core.generateExpression funcExpr
            let! args = mapM Core.generateExpression argExprs
            
            match funcExpr with
            | SynExpr.Ident(ident) ->
                return! Functions.call ident.idText args MLIRTypes.void_  // TODO: Type inference
            | _ ->
                return! Functions.call (Core.ssaOf func) args MLIRTypes.void_
        }
    
    /// Generate regular function call
    let generateFunctionCall (funcName: string) (argExpr: SynExpr): MLIRCombinator<MLIRValue> =
        mlir {
            let! arg = Core.generateExpression argExpr
            return! Functions.call funcName [arg] MLIRTypes.void_  // TODO: Look up actual return type
        }

/// Binary and unary operations
module Operations =
    
    /// Detect and generate binary operations from application syntax
    let tryGenerateBinaryOp (funcExpr: SynExpr) (leftExpr: SynExpr) (rightExpr: SynExpr): MLIRCombinator<MLIRValue option> =
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
    let generateUnaryOp (operator: string) (operand: SynExpr): MLIRCombinator<MLIRValue> =
        mlir {
            let! operandValue = Core.generateExpression operand
            return! Registry.generateOperator operator [operandValue]
        }

/// Data structure construction
module Construction =
    
    /// Generate tuple construction
    let generateTuple (exprs: SynExpr list): MLIRCombinator<MLIRValue> =
        mlir {
            let componentExprs = exprs |> List.map snd  // Extract expressions from SynTupleExpr
            let! values = mapM Core.generateExpression componentExprs
            let types = values |> List.map Core.typeOf
            let tupleType = MLIRTypes.struct_ types
            
            return! StructTypes.createRecord tupleType values
        }
    
    /// Generate array construction
    let generateArray (exprs: SynExpr list): MLIRCombinator<MLIRValue> =
        mlir {
            let! values = mapM Core.generateExpression exprs
            
            match values with
            | [] ->
                // Empty array
                let! size = Constants.intConstant 0 32
                return! Memory.alloca MLIRTypes.i32 (Some size)
                
            | firstValue :: _ ->
                let elementType = Core.typeOf firstValue
                let arraySize = values.Length
                let! size = Constants.intConstant arraySize 32
                let! arrayPtr = Memory.alloca elementType (Some size)
                
                // Initialize array elements
                for i, value in List.indexed values do
                    let! index = Constants.intConstant i 32
                    do! Memory.store value arrayPtr [index]
                
                return arrayPtr
        }
    
    /// Generate record construction
    let generateRecord (fields: SynExprRecordField list): MLIRCombinator<MLIRValue> =
        mlir {
            let extractField (field: SynExprRecordField) =
                match field with
                | SynExprRecordField(fieldName, _, exprOpt, _) ->
                    match exprOpt with
                    | Some expr -> Core.generateExpression expr
                    | None -> fail "record_field" "Missing field value"
                    
            let! fieldValues = mapM extractField fields
            
            let fieldTypes = fieldValues |> List.map Core.typeOf
            let recordType = MLIRTypes.struct_ fieldTypes
            
            return! StructTypes.createRecord recordType fieldValues
        }
    
    /// Generate anonymous record construction
    let generateAnonymousRecord (fields: (Ident * SynExpr) list): MLIRCombinator<MLIRValue> =
        mlir {
            let! fieldValues = mapM (fun (_, expr) -> Core.generateExpression expr) fields
            
            let fieldTypes = fieldValues |> List.map Core.typeOf
            let recordType = MLIRTypes.struct_ fieldTypes
            
            return! StructTypes.createRecord recordType fieldValues
        }

/// Control flow expressions
module ControlFlow =
    
    /// Generate if-then-else expression
    let generateIfThenElse (condExpr: SynExpr) (thenExpr: SynExpr) (elseExprOpt: SynExpr option): MLIRCombinator<MLIRValue> =
        mlir {
            let! condition = Core.generateExpression condExpr
            let thenBody = Core.generateExpression thenExpr
            let elseBody = 
                match elseExprOpt with
                | Some expr -> Core.generateExpression expr |>> Some
                | None -> lift None
            
            return! Conditionals.ifThenElse condition thenBody elseBody
        }
    
    /// Generate pattern match expression
    let generateMatch (expr: SynExpr) (clauses: SynMatchClause list): MLIRCombinator<MLIRValue> =
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
    let generateFor (ident: Ident) (startExpr: SynExpr) (endExpr: SynExpr) (bodyExpr: SynExpr): MLIRCombinator<MLIRValue> =
        mlir {
            let! start = Core.generateExpression startExpr
            let! end' = Core.generateExpression endExpr
            let! step = Constants.intConstant 1 32
            
            return! Loops.forLoop ident.idText start end' step (fun loopVar ->
                Core.generateExpression bodyExpr
            )
        }
    
    /// Generate for-each loop
    let generateForEach (collectionExpr: SynExpr) (bodyExpr: SynExpr): MLIRCombinator<MLIRValue> =
        mlir {
            let! collection = Core.generateExpression collectionExpr
            
            return! Loops.forEach "item" collection (fun itemVar ->
                Core.generateExpression bodyExpr
            )
        }
    
    /// Generate while loop
    let generateWhile (condExpr: SynExpr) (bodyExpr: SynExpr): MLIRCombinator<MLIRValue> =
        mlir {
            let condition = Core.generateExpression condExpr
            let body = Core.generateExpression bodyExpr
            
            return! Loops.whileLoop condition body
        }

/// Exception handling
module ExceptionHandling =
    
    /// Generate try-with expression
    let generateTryWith (tryExpr: SynExpr) (withCases: SynMatchClause list): MLIRCombinator<MLIRValue> =
        mlir {
            let tryBody = Core.generateExpression tryExpr
            
            let extractHandler (clause: SynMatchClause) =
                let (SynMatchClause(pat, _, resultExpr, _, _, _)) = clause
                (pat, Core.generateExpression resultExpr)
                
            let handlers = withCases |> List.map extractHandler
            
            return! Exceptions.tryWith tryBody handlers
        }
    
    /// Generate try-finally expression
    let generateTryFinally (tryExpr: SynExpr) (finallyExpr: SynExpr): MLIRCombinator<MLIRValue> =
        mlir {
            let tryBody = Core.generateExpression tryExpr
            let finallyBody = mlir {
                let! _ = Core.generateExpression finallyExpr
                return ()
            }
            
            return! Exceptions.tryFinally tryBody finallyBody
        }

/// Lambda expressions and closures
module Lambdas =
    
    /// Generate lambda expression
    let generateLambda (args: SynSimplePats) (bodyExpr: SynExpr): MLIRCombinator<MLIRValue> =
        mlir {
            // For now, lambdas are converted to static functions
            // Full implementation would handle closure capture
            
            let! lambdaName = nextSSA "lambda"
            do! emitComment "Lambda expression (simplified as function reference)"
            
            // TODO: Extract argument patterns and types
            // TODO: Generate function with captured variables as parameters
            
            return Core.createValue lambdaName (MLIRTypes.func [MLIRTypes.i32] MLIRTypes.i32)
        }

/// Let bindings and local definitions
module LetBindings =
    
    /// Generate let binding
    let generateLet (bindings: SynBinding list) (bodyExpr: SynExpr): MLIRCombinator<MLIRValue> =
        mlir {
            // Process each binding
            for binding in bindings do
                let (SynBinding(_, _, _, _, _, _, _, pat, _, expr, _, _, _)) = binding
                let! value = Core.generateExpression expr
                do! Bindings.bindPattern pat value
            
            // Generate body with bindings in scope
            return! Core.generateExpression bodyExpr
        }

/// Field access and mutation
module FieldAccess =
    
    /// Generate field access
    let generateFieldAccess (expr: SynExpr) (fieldIds: Ident list): MLIRCombinator<MLIRValue> =
        mlir {
            let! record = Core.generateExpression expr
            let fieldName = fieldIds |> List.map (fun id -> id.idText) |> String.concat "."
            
            // For now, assume field index based on name hash
            let fieldIndex = fieldName.GetHashCode() % 8
            
            return! StructTypes.accessField record fieldIndex
        }
    
    /// Generate field assignment
    let generateFieldSet (expr: SynExpr) (fieldIds: Ident list) (valueExpr: SynExpr): MLIRCombinator<MLIRValue> =
        mlir {
            let! record = Core.generateExpression expr
            let! newValue = Core.generateExpression valueExpr
            let fieldName = fieldIds |> List.map (fun id -> id.idText) |> String.concat "."
            
            let fieldIndex = fieldName.GetHashCode() % 8
            
            return! StructTypes.updateField record fieldIndex newValue
        }

/// Object construction
module ObjectConstruction =
    
    /// Generate object construction
    let generateNew (typeName: SynType) (argExpr: SynExpr): MLIRCombinator<MLIRValue> =
        mlir {
            let! mlirType = Mapping.synTypeToMLIR typeName
            let! arg = Core.generateExpression argExpr
            
            // For now, simple allocation
            return! Memory.alloca mlirType None
        }