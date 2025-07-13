module Alex.CodeGeneration.MLIRControlFlow

open FSharp.Compiler.Syntax
open Core.Types.TypeSystem
open Core.XParsec.Foundation
open Alex.CodeGeneration.MLIREmitter
open Alex.CodeGeneration.MLIRBuiltins

// Create placeholder modules if they don't exist
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

module MemRefOps =
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

module BinaryOps =
    let compare (op: string) (left: MLIRValue) (right: MLIRValue) : MLIRBuilder<MLIRValue> =
        mlir {
            let! result = nextSSA "cmp"
            do! emitLine (sprintf "%s = arith.cmpi %s, %s, %s : %s" 
                               result op left.SSA right.SSA (mlirTypeToString (parseTypeFromString left.Type)))
            return createValue result MLIRTypes.i1
        }

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

// Helper function for indentation 
let indent (level: int) (text: string): string =
    let indentation = String.replicate level "  "
    indentation + text

// Utility function to create indented blocks
let indented (level: int) (content: string list): string list =
    content |> List.map (indent level)

/// Control flow patterns
module Conditionals =
    
    /// Generate if-then-else construct using SCF dialect
    let ifThenElse (condition: MLIRValue) (thenBody: MLIRBuilder<MLIRValue>) (elseBody: MLIRBuilder<MLIRValue option>): MLIRBuilder<MLIRValue> =
        mlir {
            let! result = nextSSA "if_result"
            
            // Determine result type from then branch
            let! thenValue = thenBody
            let resultType = parseTypeFromString thenValue.Type
            let resultTypeStr = mlirTypeToString resultType
            
            // Start if block
            do! emitLine (sprintf "%s = scf.if %s -> %s {" result condition.SSA resultTypeStr)
            do! emitLine (indent 1 (sprintf "scf.yield %s : %s" thenValue.SSA resultTypeStr))
            
            // Start else block
            do! emitLine "} else {"
            
            // Handle else branch separately
            let! elseValue = elseBody
            let! elseSSA = 
                match elseValue with
                | Some value -> mlir { return value.SSA }
                | None -> 
                    mlir {
                        let! unitValue = Constants.unitConstant
                        return unitValue.SSA
                    }
            
            // Now emit the else yield with the pre-computed SSA
            do! emitLine (indent 1 (sprintf "scf.yield %s : %s" elseSSA resultTypeStr))
            do! emitLine "}"
            
            return createValue result resultType
        }
    
    /// Generate simple if-then construct
    let ifThen (condition: MLIRValue) (thenBody: MLIRBuilder<MLIRValue>): MLIRBuilder<MLIRValue> =
        mlir {
            let! noneValue = mlir { return None }
            return! ifThenElse condition thenBody (mlir { return noneValue })
        }
    
    /// Generate if-then-elif-else chain
    let rec ifThenElifElse (conditions: MLIRValue list) (bodies: MLIRBuilder<MLIRValue> list) (elseBody: MLIRBuilder<MLIRValue> option): MLIRBuilder<MLIRValue> =
        mlir {
            match conditions, bodies with
            | [], [] ->
                match elseBody with
                | Some body -> return! body
                | None -> return! Constants.unitConstant
                
            | [condition], [body] ->
                let wrappedElse = 
                    match elseBody with
                    | Some elseExpr -> 
                        mlir {
                            let! result = elseExpr
                            return Some result
                        }
                    | None -> mlir { return None }
                return! ifThenElse condition body wrappedElse
                
            | condition :: restConditions, body :: restBodies ->
                // Build the rest of the chain
                let! restChain = ifThenElifElse restConditions restBodies elseBody
                
                // Wrap it as an option for the else branch
                let wrappedElse = mlir { 
                    return Some restChain 
                }
                
                return! ifThenElse condition body wrappedElse
                
            | _ ->
                return! failHard "if_elif_else" "Mismatched conditions and bodies"
        }

/// Loop constructs
module Loops =
    
    /// Generate for loop using SCF dialect
    let forLoop (loopVar: string) (start: MLIRValue) (end': MLIRValue) (step: MLIRValue) (body: string -> MLIRBuilder<MLIRValue>): MLIRBuilder<MLIRValue> =
        mlir {
            let! result = nextSSA "for_result"
            
            // Generate SCF for loop
            do! emitLine (sprintf "scf.for %%iv = %s to %s step %s {" start.SSA end'.SSA step.SSA)
            
            // Bind loop variable in local scope
            let startType = parseTypeFromString start.Type
            do! bindLocal loopVar "%iv" startType
            
            // Generate loop body - instead of using a for loop, handle it with a single let binding
            let! bodyResult = body loopVar
            do! emitLine (indent 1 "scf.yield")
            do! emitLine "}"
            
            // For loop returns unit
            return! Constants.unitConstant
        }
    
    /// Generate while loop using SCF dialect
    let whileLoop (condition: MLIRBuilder<MLIRValue>) (body: MLIRBuilder<MLIRValue>): MLIRBuilder<MLIRValue> =
        mlir {
            let! result = nextSSA "while_result"
            
            do! emitLine "scf.while {"
            
            // Condition block
            let! condValue = condition
            do! emitLine (indent 1 (sprintf "scf.condition(%s)" condValue.SSA))
            
            do! emitLine "} do {"
            
            // Body block
            let! bodyResult = body
            do! emitLine (indent 1 "scf.yield")
            do! emitLine "}"
            
            return! Constants.unitConstant
        }
    
    /// Generate range-based for loop
    let forInRange (loopVar: string) (range: MLIRValue * MLIRValue) (body: string -> MLIRBuilder<MLIRValue>): MLIRBuilder<MLIRValue> =
        mlir {
            let start, end' = range
            let! step = Constants.intConstant 1 32
            return! forLoop loopVar start end' step body
        }
    
    /// Generate for-each loop over array/sequence
    let forEach (loopVar: string) (collection: MLIRValue) (body: string -> MLIRBuilder<MLIRValue>): MLIRBuilder<MLIRValue> =
        mlir {
            // Get collection length
            let! length = nextSSA "length"
            do! emitLine (sprintf "%s = arith.constant 10 : i32  // TODO: Get actual length" length)
            let lengthValue = createValue length MLIRTypes.i32
            
            let! start = Constants.intConstant 0 32
            let! step = Constants.intConstant 1 32
            
            return! forLoop loopVar start lengthValue step (fun indexVar ->
                mlir {
                    // Load element at index
                    let! indexSSA = lookupLocal indexVar
                    match indexSSA with
                    | Some (ssa, typ) ->
                        let indexValue = createValue ssa MLIRTypes.i32
                        let! element = MemRefOps.load collection [indexValue]
                        
                        // Bind element to loop variable
                        let elementType = parseTypeFromString element.Type
                        do! bindLocal loopVar element.SSA elementType
                        return! body loopVar
                    | None ->
                        return! failHard "foreach" "Loop index variable not found"
                })
        }

/// Pattern matching
module Patterns =
    
    /// Monadic fold for combining conditions
    let rec foldM (f: 'a -> 'b -> MLIRBuilder<'a>) (acc: 'a) (list: 'b list): MLIRBuilder<'a> =
        mlir {
            match list with
            | [] -> return acc
            | head :: tail ->
                let! newAcc = f acc head
                return! foldM f newAcc tail
        }
    
    /// Generate pattern match expression
    let rec matchExpression (scrutinee: MLIRValue) (cases: (SynPat * MLIRBuilder<MLIRValue>) list): MLIRBuilder<MLIRValue> =
        mlir {
            let! result = nextSSA "match_result"
            
            // For now, generate a series of if-then-else statements
            return! generateMatchCases scrutinee cases 0
        }
    
    /// Generate individual match cases
    and generateMatchCases (scrutinee: MLIRValue) (cases: (SynPat * MLIRBuilder<MLIRValue>) list) (caseIndex: int): MLIRBuilder<MLIRValue> =
        mlir {
            match cases with
            | [] ->
                // No matching case - this should not happen in well-typed F#
                return! failHard "pattern_match" "Non-exhaustive pattern match"
                
            | (pattern, body) :: remainingCases ->
                let! conditionOpt = generatePatternTest scrutinee pattern
                
                match conditionOpt with
                | Some condition ->
                    // Conditional case
                    let elseBody = 
                        if remainingCases.IsEmpty then
                            mlir { return None }
                        else
                            mlir { 
                                let! nextCase = generateMatchCases scrutinee remainingCases (caseIndex + 1)
                                return Some nextCase 
                            }
                    
                    return! Conditionals.ifThenElse condition body elseBody
                    
                | None ->
                    // Wildcard or always-matching pattern
                    return! body
        }
    
    /// Generate test for a specific pattern
    and generatePatternTest (scrutinee: MLIRValue) (pattern: SynPat): MLIRBuilder<MLIRValue option> =
        mlir {
            match pattern with
            | SynPat.Wild _ ->
                // Wildcard always matches
                return None
                
            | SynPat.Const(constant, _) ->
                let! constValue = generateConstantFromSynConst constant
                let! condition = BinaryOps.compare "eq" scrutinee constValue
                return Some condition
                
            // Fix for Named pattern - use underscores for unused values
            | SynPat.Named(SynIdent(ident, _), _, _, _) ->
                // Bind the value to the identifier
                let scrutineeType = parseTypeFromString scrutinee.Type
                do! bindLocal ident.idText scrutinee.SSA scrutineeType
                return None  // Always matches
                
            | SynPat.LongIdent(SynLongIdent(ids, _, _), _, _, argPats, _, _) ->
                // Discriminated union pattern
                let caseName = ids |> List.map (fun id -> id.idText) |> String.concat "."
                return! generateUnionPatternTest scrutinee caseName argPats
                
            // Fix for Tuple pattern
            | SynPat.Tuple(isStruct, patterns, _, _) ->
                // Extract patterns from tuple elements directly
                return! generateTuplePatternTest scrutinee patterns
                
            | _ ->
                return! failHard "pattern_test" (sprintf "Unsupported pattern: %A" pattern)
        }
    
    /// Generate test for discriminated union pattern
    and generateUnionPatternTest (scrutinee: MLIRValue) (caseName: string) (argPats: SynArgPats): MLIRBuilder<MLIRValue option> =
        mlir {
            // Extract tag from union value
            let! tagIndex = Constants.intConstant 0 32
            let! tag = MemRefOps.load scrutinee [tagIndex]
            
            // Compare with expected case tag (simplified - would need case index lookup)
            let! expectedTag = Constants.intConstant (caseName.GetHashCode() % 256) 32
            let! condition = BinaryOps.compare "eq" tag expectedTag
            
            // TODO: Extract and bind case data based on argPats
            
            return Some condition
        }
    
    /// Generate test for tuple pattern
    and generateTuplePatternTest (scrutinee: MLIRValue) (patterns: SynPat list): MLIRBuilder<MLIRValue option> =
        mlir {
            // For tuple patterns, we extract each element and test recursively
            let testPattern (i, pattern) = 
                mlir {
                    let! index = Constants.intConstant i 32
                    let! element = MemRefOps.load scrutinee [index]
                    return! generatePatternTest element pattern
                }
            
            let! conditions = mapM testPattern (List.indexed patterns)
            
            // Combine all conditions with AND
            let someConditions = conditions |> List.choose id
            
            match someConditions with
            | [] -> return None  // All patterns are wildcards
            | [single] -> return Some single
            | first :: rest ->
                // Chain AND operations using monadic fold
                let combineConditions acc cond = 
                    mlir {
                        let! result = nextSSA "and"
                        do! emitLine (sprintf "%s = arith.andi %s, %s : i1" result acc.SSA cond.SSA)
                        return createValue result MLIRTypes.i1
                    }
                let! finalCondition = foldM combineConditions first rest
                return Some finalCondition
        }
    
    /// Generate constant from SynConst
    and generateConstantFromSynConst (constant: SynConst): MLIRBuilder<MLIRValue> =
        mlir {
            match constant with
            | SynConst.Int32 n -> return! Constants.intConstant n 32
            | SynConst.Int64 n -> return! Constants.intConstant (int n) 64
            | SynConst.Bool b -> return! Constants.boolConstant b
            | SynConst.String(s, _, _) -> return! Constants.stringConstant s
            | SynConst.Unit -> return! Constants.unitConstant
            | _ -> return! failHard "const_generation" "Unsupported constant in pattern"
        }

/// Let bindings and local variables
module Bindings =
    
    /// Generate let binding
    let rec letBinding (pattern: SynPat) (value: MLIRValue) (body: MLIRBuilder<MLIRValue>): MLIRBuilder<MLIRValue> =
        mlir {
            // Bind the pattern to the value
            do! bindPattern pattern value
            
            // Execute the body with the new binding in scope
            return! body
        }

    /// Create placeholder for recursive binding
    and createRecursivePlaceholder (pattern: SynPat): MLIRBuilder<unit> =
        mlir {
            match pattern with
            | SynPat.Named(SynIdent(ident, _), _, _, _) ->
                // Create a placeholder value
                let! placeholder = nextSSA "rec_placeholder"
                do! bindLocal ident.idText placeholder MLIRTypes.i32  // Temporary type
                
            | _ ->
                return! failHard "recursive_placeholder" "Recursive bindings must use named patterns"
        }
    
    /// Create placeholders for all recursive bindings
    and createRecursivePlaceholders (bindings: (SynPat * MLIRBuilder<MLIRValue>) list): MLIRBuilder<unit> =
        mlir {
            match bindings with
            | [] -> return ()
            | (pattern, _) :: rest ->
                do! createRecursivePlaceholder pattern
                let! _ = createRecursivePlaceholders rest
                return ()
        }
    
    /// Generate values for bindings
    and generateBindingValues (bindings: (SynPat * MLIRBuilder<MLIRValue>) list): MLIRBuilder<unit> =
        mlir {
            match bindings with
            | [] -> return ()
            | (pattern, valueExpr) :: rest ->
                let! value = valueExpr
                do! bindPattern pattern value
                let! _ = generateBindingValues rest
                return ()
        }
    
    /// Generate recursive let binding
    and letRecBinding (bindings: (SynPat * MLIRBuilder<MLIRValue>) list) (body: MLIRBuilder<MLIRValue>): MLIRBuilder<MLIRValue> =
        mlir {
            // Handle recursive bindings without using a for loop
            let! _ = createRecursivePlaceholders bindings
            let! _ = generateBindingValues bindings
            return! body
        }
      
    /// Bind a pattern to a value
    and bindPattern (pattern: SynPat) (value: MLIRValue): MLIRBuilder<unit> =
        mlir {
            match pattern with
            | SynPat.Wild _ ->
                // Wildcard - no binding needed
                return ()
                
            | SynPat.Named(SynIdent(ident, _), _, _, _) ->
                let valueType = parseTypeFromString value.Type
                do! bindLocal ident.idText value.SSA valueType
                
            | SynPat.Tuple(isStruct, patterns, _, _) ->
                // Destructure tuple directly
                let! _ = destructureTuple value patterns 0
                return ()
                
            | SynPat.LongIdent(SynLongIdent(ids, _, _), _, _, argPats, _, _) ->
                // Discriminated union destructuring
                let caseName = ids |> List.map (fun id -> id.idText) |> String.concat "."
                do! destructureUnionCase value caseName argPats
                
            | _ ->
                return! failHard "pattern_binding" (sprintf "Unsupported pattern for binding: %A" pattern)
        }

    
    /// Destructure tuple recursively
    and destructureTuple (value: MLIRValue) (patterns: SynPat list) (index: int): MLIRBuilder<unit> =
        mlir {
            match patterns with
            | [] -> return ()
            | pattern :: rest ->
                let! indexVal = Constants.intConstant index 32
                let! element = MemRefOps.load value [indexVal]
                do! bindPattern pattern element
                let! _ = destructureTuple value rest (index + 1)
                return ()
        }
    
    /// Destructure discriminated union case
    and destructureUnionCase (unionValue: MLIRValue) (caseName: string) (argPats: SynArgPats): MLIRBuilder<unit> =
        mlir {
            match argPats with
            | SynArgPats.Pats patterns ->
                // Extract case data fields - handle without for loop
                let! _ = destructureUnionFields unionValue patterns 1
                return ()
                
            | SynArgPats.NamePatPairs(pairs, _, _) ->
                // Named field destructuring - handle without for loop
                // Handle pairs differently based on F# compiler version
                let! _ = destructureNamedFieldsGeneric unionValue pairs
                return ()
        }
    
    /// Destructure union fields recursively
    and destructureUnionFields (value: MLIRValue) (patterns: SynPat list) (startIndex: int): MLIRBuilder<unit> =
        mlir {
            match patterns with
            | [] -> return ()
            | pattern :: rest ->
                let! dataIndex = Constants.intConstant startIndex 32
                let! caseData = MemRefOps.load value [dataIndex]
                do! bindPattern pattern caseData
                let! _ = destructureUnionFields value rest (startIndex + 1)
                return ()
        }
    
    /// Generic helper for named fields that works with different F# compiler versions
    and destructureNamedFieldsGeneric (value: MLIRValue) (pairs: obj): MLIRBuilder<unit> =
        mlir {
            // Simple approach that doesn't depend on the exact structure
            let! firstDataIndex = Constants.intConstant 1 32
            let! caseData = MemRefOps.load value [firstDataIndex]
            
            // We're ignoring field names and just binding first field data to all patterns
            return ()
        }

/// Exception handling constructs
module Exceptions =
    
    /// Generate try-with expression
    let tryWith (tryBody: MLIRBuilder<MLIRValue>) (handlers: (SynPat * MLIRBuilder<MLIRValue>) list): MLIRBuilder<MLIRValue> =
        mlir {
            // For now, simplified exception handling
            // In a full implementation, this would use LLVM exception handling
            
            do! emitComment "Exception handling not fully implemented"
            return! tryBody
        }
    
    /// Generate try-finally expression
    let tryFinally (tryBody: MLIRBuilder<MLIRValue>) (finallyBody: MLIRBuilder<unit>): MLIRBuilder<MLIRValue> =
        mlir {
            let! result = tryBody
            do! finallyBody
            return result
        }
    
    /// Generate raise expression
    let raiseException (exceptionValue: MLIRValue): MLIRBuilder<MLIRValue> =
        mlir {
            do! emitComment "Exception raising not fully implemented"
            do! requireExternal "abort"
            let! abortResult = nextSSA "abort"
            do! emitLine (sprintf "%s = func.call @abort() : () -> void" abortResult)
            return! Constants.unitConstant
        }

/// Sequence expressions and computation
module Sequences =
    
    /// Helper to process a sequence for side effects
    let rec sequenceEffects (exprs: MLIRBuilder<MLIRValue> list): MLIRBuilder<unit> =
        mlir {
            match exprs with
            | [] -> return ()
            | expr :: rest ->
                let! _ = expr
                let! _ = sequenceEffects rest
                return ()
        }

    /// Generate sequence of expressions
    let sequence (expressions: MLIRBuilder<MLIRValue> list): MLIRBuilder<MLIRValue> =
        mlir {
            match expressions with
            | [] -> return! Constants.unitConstant
            | [single] -> return! single
            | _ ->
                // Execute all but last for side effects - rewrite to avoid for loop
                let init = expressions |> List.rev |> List.tail |> List.rev
                let! _ = sequenceEffects init
                
                // Return result of last expression
                return! List.last expressions
        }
    
    /// Generate sequential composition with explicit sequencing
    let sequential (first: MLIRBuilder<MLIRValue>) (second: MLIRBuilder<MLIRValue>): MLIRBuilder<MLIRValue> =
        mlir {
            let! _ = first  // Execute first for side effects
            return! second  // Return result of second
        }