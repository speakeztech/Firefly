module Alex.CodeGeneration.MLIRModuleGenerator

open System.IO
open FSharp.Compiler.Syntax
open Core.Types.TypeSystem
open Alex.Analysis.CompilationUnit
open Alex.Bindings.SymbolRegistry
open TypeMapping
open MLIREmitter
open MLIRBuiltins

// ===================================================================
// AST Pattern Combinators - XParsec-style Pattern Matching
// ===================================================================

/// AST parser result type for backtracking
type ASTResult<'T> = 'T option

/// AST combinator - transforms AST nodes with MLIRBuilder for effects
type ASTCombinator<'AST, 'Result> = 'AST -> MLIRBuilder<ASTResult<'Result>>

/// Lift a pure pattern match into an AST combinator
let pattern (f: 'AST -> 'Result option) : ASTCombinator<'AST, 'Result> =
    fun ast -> mlir.Return (f ast)

/// Sequential bind for AST combinators
let (>>=%) (p: ASTCombinator<'AST, 'A>) (f: 'A -> MLIRBuilder<'B>) : ASTCombinator<'AST, 'B> =
    fun ast ->
        mlir {
            let! resultOpt = p ast
            match resultOpt with
            | Some a -> 
                let! b = f a
                return Some b
            | None -> return None
        }

/// Choice between two AST combinators (backtracking)
let (<|>) (p1: ASTCombinator<'AST, 'Result>) (p2: ASTCombinator<'AST, 'Result>) : ASTCombinator<'AST, 'Result> =
    fun ast ->
        mlir {
            let! result1 = p1 ast
            match result1 with
            | Some _ -> return result1
            | None -> return! p2 ast
        }

/// Map over AST combinator result
let (|>>) (p: ASTCombinator<'AST, 'A>) (f: 'A -> 'B) : ASTCombinator<'AST, 'B> =
    fun ast ->
        mlir {
            let! resultOpt = p ast
            return Option.map f resultOpt
        }

/// Try multiple combinators until one succeeds
let choice (parsers: ASTCombinator<'AST, 'Result> list) : ASTCombinator<'AST, 'Result> =
    List.reduce (<|>) parsers

/// Always succeed with a value
let preturn (value: 'Result) : ASTCombinator<'AST, 'Result> =
    fun _ -> mlir.Return (Some value)

/// Always fail (for backtracking)
let pfail<'AST, 'Result> : ASTCombinator<'AST, 'Result> =
    fun _ -> mlir.Return None

/// Emit comment helper
let emitComment (comment: string) : MLIRBuilder<unit> =
    emitLine (sprintf "// %s" comment)

// ===================================================================
// Type-Preserving Expression Patterns
// ===================================================================

/// Pattern for integer constants with type
let pInt32 : ASTCombinator<SynExpr, int * MLIRType> =
    pattern (function
        | SynExpr.Const(SynConst.Int32 n, _) -> Some (n, MLIRTypes.i32)
        | _ -> None)

let pInt64 : ASTCombinator<SynExpr, int64 * MLIRType> =
    pattern (function
        | SynExpr.Const(SynConst.Int64 n, _) -> Some (n, MLIRTypes.i64)
        | _ -> None)

let pFloat : ASTCombinator<SynExpr, float * MLIRType> =
    pattern (function
        | SynExpr.Const(SynConst.Double f, _) -> Some (f, MLIRTypes.f64)
        | _ -> None)

let pFloat32 : ASTCombinator<SynExpr, float32 * MLIRType> =
    pattern (function
        | SynExpr.Const(SynConst.Single f, _) -> Some (f, MLIRTypes.f32)
        | _ -> None)

let pBool : ASTCombinator<SynExpr, bool * MLIRType> =
    pattern (function
        | SynExpr.Const(SynConst.Bool b, _) -> Some (b, MLIRTypes.i1)
        | _ -> None)

let pString : ASTCombinator<SynExpr, string * MLIRType> =
    pattern (function
        | SynExpr.Const(SynConst.String(s, _, _), _) -> Some (s, MLIRTypes.memref MLIRTypes.i8)
        | _ -> None)

let pUnit : ASTCombinator<SynExpr, MLIRType> =
    pattern (function
        | SynExpr.Const(SynConst.Unit, _) -> Some MLIRTypes.void_
        | _ -> None)

let pIdent : ASTCombinator<SynExpr, string> =
    pattern (function
        | SynExpr.Ident ident -> Some ident.idText
        | SynExpr.LongIdent(_, SynLongIdent(longDotId, _, _), _, _) ->
            let path = longDotId |> List.map (fun id -> id.idText)
            Some (String.concat "." path)
        | _ -> None)

// ===================================================================
// MLIR Value Generation with Type Preservation
// ===================================================================

/// Generate typed constant
let genConst (value: obj) (typ: MLIRType) : MLIRBuilder<MLIRValue> =
    mlir {
        let valueStr = 
            match value with
            | :? bool as b -> if b then "1" else "0"
            | :? string as s -> sprintf "\"%s\"" s
            | _ -> string value
        
        let! ssa = nextSSA "const"
        let typeStr = mlirTypeToString typ
        do! emitLine (sprintf "%s = arith.constant %s : %s" ssa valueStr typeStr)
        return createValue ssa typ
    }

/// Generate identifier lookup
let genIdentRef (name: string) : MLIRBuilder<MLIRValue> =
    mlir {
        let! state = getState
        match Map.tryFind name state.LocalVars with
        | Some (ssa, typ) ->
            return createValue ssa typ
        | None ->
            // Generate undefined reference with error comment
            do! emitComment (sprintf "Undefined identifier: %s" name)
            return createValue "%undefined" MLIRTypes.i32
    }

// ===================================================================
// Composable Expression Parser Using XParsec Patterns
// ===================================================================

/// Parse constants and generate MLIR values
let pConstant : ASTCombinator<SynExpr, MLIRValue> =
    choice [
        pInt32 >>=% fun (n, typ) -> genConst n typ
        pInt64 >>=% fun (n, typ) -> genConst n typ
        pFloat >>=% fun (f, typ) -> genConst f typ
        pFloat32 >>=% fun (f, typ) -> genConst f typ
        pBool >>=% fun (b, typ) -> genConst b typ
        pString >>=% fun (s, typ) -> genConst s typ
        pUnit >>=% fun typ -> genConst () typ
    ]

/// Parse identifier references
let pIdentExpr : ASTCombinator<SynExpr, MLIRValue> =
    pIdent >>=% genIdentRef

/// Parse function application
let pApp : ASTCombinator<SynExpr, (SynExpr * SynExpr)> =
    pattern (function
        | SynExpr.App(_, _, funcExpr, argExpr, _) -> Some (funcExpr, argExpr)
        | _ -> None)

/// Parse if-then-else expression
let pIfThenElse : ASTCombinator<SynExpr, (SynExpr * SynExpr * SynExpr option)> =
    pattern (function
        | SynExpr.IfThenElse(cond, thenExpr, elseExpr, _, _, _, _) -> Some (cond, thenExpr, elseExpr)
        | _ -> None)

/// Parse any expression (main entry point)
let rec pExpression : ASTCombinator<SynExpr, MLIRValue> =
    choice [
        pConstant
        pIdentExpr
        pApplicationExpr
        pIfThenElseExpr
        pFallback
    ]

and pApplicationExpr : ASTCombinator<SynExpr, MLIRValue> =
    pApp >>=% fun (funcExpr, argExpr) ->
        mlir {
            let! funcOpt = pIdent funcExpr
            let! argOpt = pExpression argExpr
            
            match funcOpt, argOpt with
            | Some funcName, Some arg ->
                let! result = nextSSA "call"
                // For now, assume i32 return type - would need type inference/lookup
                do! emitLine (sprintf "%s = func.call @%s(%s) : (%s) -> i32" 
                                result funcName arg.SSA arg.Type)
                return createValue result MLIRTypes.i32
            | _ ->
                do! emitComment "Failed to parse function application"
                return createValue "%error" MLIRTypes.i32
        }

and pIfThenElseExpr : ASTCombinator<SynExpr, MLIRValue> =
    pIfThenElse >>=% fun (condExpr, thenExpr, elseOpt) ->
        mlir {
            let! condOpt = pExpression condExpr
            match condOpt with
            | Some cond ->
                let! result = nextSSA "if_result"
                let! thenLabel = nextSSA "then"
                let! elseLabel = nextSSA "else"
                let! mergeLabel = nextSSA "merge"
                
                // Conditional branch
                do! emitLine (sprintf "cf.cond_br %s, ^%s, ^%s" cond.SSA thenLabel elseLabel)
                
                // Then block
                do! emitLine (sprintf "^%s:" thenLabel)
                do! updateState (fun s -> { s with Indent = s.Indent + 1 })
                let! thenOpt = pExpression thenExpr
                do! match thenOpt with
                    | Some thenVal ->
                        emitLine (sprintf "cf.br ^%s(%s : %s)" mergeLabel thenVal.SSA thenVal.Type)
                    | None ->
                        mlir {
                            do! emitComment "Failed to parse then branch"
                            do! emitLine (sprintf "cf.br ^%s" mergeLabel)
                        }
                do! updateState (fun s -> { s with Indent = s.Indent - 1 })
                
                // Else block
                do! emitLine (sprintf "^%s:" elseLabel)
                do! updateState (fun s -> { s with Indent = s.Indent + 1 })
                do! match elseOpt with
                    | Some elseExpr ->
                        mlir {
                            let! elseValOpt = pExpression elseExpr
                            do! match elseValOpt with
                                | Some elseVal ->
                                    emitLine (sprintf "cf.br ^%s(%s : %s)" mergeLabel elseVal.SSA elseVal.Type)
                                | None ->
                                    mlir {
                                        do! emitComment "Failed to parse else branch"
                                        do! emitLine (sprintf "cf.br ^%s" mergeLabel)
                                    }
                        }
                    | None ->
                        emitLine (sprintf "cf.br ^%s" mergeLabel)
                do! updateState (fun s -> { s with Indent = s.Indent - 1 })
                
                // Merge block
                do! emitLine (sprintf "^%s(%s: i32):" mergeLabel result)
                return createValue result MLIRTypes.i32
            | None ->
                do! emitComment "Failed to parse condition"
                return createValue "%error" MLIRTypes.i32
        }

and pFallback : ASTCombinator<SynExpr, MLIRValue> =
    fun expr ->
        mlir {
            do! emitComment (sprintf "Unsupported expression: %A" (expr.GetType().Name))
            return Some (createValue "%unsupported" MLIRTypes.i32)
        }

// ===================================================================
// Function and Type Definition Patterns
// ===================================================================

/// Pattern for function bindings
let pFunctionBinding : ASTCombinator<SynBinding, string * MLIRType list * MLIRType> =
    pattern (function
        | SynBinding(_, _, _, _, _, _, valData, SynPat.Named(SynIdent(ident, _), _, _, _), returnInfo, _, _, _, _) ->
            let name = ident.idText
            
            // Extract parameter types (simplified)
            let paramTypes = 
                let (SynValData(_, SynValInfo(paramGroups, _), _)) = valData
                paramGroups |> List.concat |> List.map (fun _ -> MLIRTypes.i32)
            
            // Extract return type
            let returnType =
                match returnInfo with
                | Some(SynBindingReturnInfo(retType, _, _, _)) ->
                    // Simplified type conversion
                    MLIRTypes.i32
                | None -> MLIRTypes.void_
                
            Some (name, paramTypes, returnType)
        | _ -> None)

/// Generate function with body
let genFunction (binding: SynBinding) (name: string) (paramTypes: MLIRType list) (returnType: MLIRType) : MLIRBuilder<unit> =
    let (SynBinding(_, _, _, _, _, _, _, _, _, expr, _, _, _)) = binding
    
    mlir {
        // Generate function declaration
        let parameters = paramTypes |> List.mapi (fun i t -> (sprintf "arg%d" i, t))
        let retTypes = if returnType = MLIRTypes.void_ then [] else [returnType]
        
        let paramStr = parameters |> List.map (fun (n, t) -> sprintf "%%%s: %s" n (mlirTypeToString t)) |> String.concat ", "
        let returnStr = 
            match retTypes with
            | [] -> ""
            | types -> " -> " + (types |> List.map mlirTypeToString |> String.concat ", ")
        
        do! emitLine (sprintf "func.func @%s(%s)%s {" name paramStr returnStr)
        
        // Generate body with proper scope
        do! updateState (fun s -> { s with Indent = s.Indent + 1 })
        let! resultOpt = pExpression expr
        do! match resultOpt with
            | Some value ->
                if returnType = MLIRTypes.void_ then
                    emitLine "func.return"
                else
                    emitLine (sprintf "func.return %s : %s" value.SSA value.Type)
            | None ->
                mlir {
                    do! emitComment "Failed to parse function body"
                    do! emitLine "func.return"
                }
        do! updateState (fun s -> { s with Indent = s.Indent - 1 })
        
        do! emitLine "}"
    }

/// Parse and generate a complete function
let pFunction : ASTCombinator<SynBinding, unit> =
    fun binding ->
        mlir {
            let! resultOpt = pFunctionBinding binding
            match resultOpt with
            | Some (name, paramTypes, returnType) ->
                do! genFunction binding name paramTypes returnType
                return Some ()
            | None ->
                do! emitComment "Unsupported binding pattern"
                return None
        }

// ===================================================================
// Module-Level Patterns
// ===================================================================

/// Pattern for module declarations
let rec pModuleDecl : ASTCombinator<SynModuleDecl, unit> =
    pattern (fun decl -> Some decl) >>=% fun decl ->
        match decl with
        | SynModuleDecl.Let(_, bindings, _) ->
            mlir {
                do! emitComment "Module-level bindings"
                // Process bindings recursively
                let rec processBindings bs =
                    mlir {
                        match bs with
                        | [] -> return ()
                        | binding::rest ->
                            let! _ = pFunction binding
                            return! processBindings rest
                    }
                do! processBindings bindings
                return ()
            }
            
        | SynModuleDecl.Types(typeDefns, _) ->
            mlir {
                do! emitComment "Type definitions"
                return ()
            }
            
        | SynModuleDecl.NestedModule(componentInfo, _, decls, _, _, _) ->
            mlir {
                let (SynComponentInfo(_, _, _, longId, _, _, _, _)) = componentInfo
                let moduleName = String.concat "." (longId |> List.map (fun id -> id.idText))
                do! emitComment (sprintf "Nested module: %s" moduleName)
                // Process declarations recursively
                let rec processDecls ds =
                    mlir {
                        match ds with
                        | [] -> return ()
                        | decl::rest ->
                            let! _ = pModuleDecl decl
                            return! processDecls rest
                    }
                do! processDecls decls
                return ()
            }
            
        | _ ->
            mlir {
                do! emitComment "Unsupported module declaration"
                return ()
            }

// ===================================================================
// Main Entry Point - XParsec-based MLIR Generation
// ===================================================================

/// Generate MLIR module from parsed input
let generateMLIR (parsedInput: ParsedInput) : MLIRBuilder<unit> =
    mlir {
        do! emitLine "module {"
        
        do! updateState (fun s -> { s with Indent = s.Indent + 1 })
        do! match parsedInput with
            | ParsedInput.ImplFile(ParsedImplFileInput(fileName, _, _, _, _, modules, _, _, _)) ->
                mlir {
                    do! emitComment (sprintf "File: %s" fileName)
                    
                    // Process modules recursively
                    let rec processModules mods =
                        mlir {
                            match mods with
                            | [] -> return ()
                            | (SynModuleOrNamespace(longId, _, _, decls, _, _, _, _, _))::rest ->
                                let namespaceName = String.concat "." (longId |> List.map (fun id -> id.idText))
                                do! emitComment (sprintf "Module/namespace: %s" namespaceName)
                                
                                // Process declarations recursively
                                let rec processDecls ds =
                                    mlir {
                                        match ds with
                                        | [] -> return ()
                                        | d::drest ->
                                            let! _ = pModuleDecl d
                                            return! processDecls drest
                                    }
                                do! processDecls decls
                                return! processModules rest
                        }
                    do! processModules modules
                }
                
            | ParsedInput.SigFile _ ->
                emitComment "Signature files not implemented"
        do! updateState (fun s -> { s with Indent = s.Indent - 1 })
                
        do! emitLine "}"
    }

/// Run the MLIR generator and extract result
let generateModuleFromAST (ast: ParsedInput) (typeCtx: TypeMapping.TypeContext) (symbolRegistry: Alex.Bindings.SymbolRegistry.SymbolRegistry) : string =
    let initialState = createInitialState typeCtx symbolRegistry
    
    let (_, finalState) = runBuilder (generateMLIR ast) initialState
    finalState.Output.ToString()

/// Generate MLIR from analyzed compilation unit
let generateFromCompilationAnalysis (analysis: CompilationUnitAnalysis) (typeCtx: TypeContext) (symbolRegistry: SymbolRegistry) : string =
    // Get the main file's pruned AST
    let mainFile = analysis.Unit.MainFile
    match Map.tryFind mainFile analysis.PrunedAsts with
    | None -> failwith (sprintf "Main file %s not found in pruned ASTs" mainFile)
    | Some mainAst ->
        // Generate MLIR with full context awareness
        // The AST has already been pruned based on reachability
        let initialState = createInitialState typeCtx symbolRegistry
        let (_, finalState) = runBuilder (generateMLIR mainAst) initialState
        finalState.Output.ToString()

/// Generate MLIR for multiple files with cross-file awareness
let generateMultiFileMLIR (analysis: CompilationUnitAnalysis) (typeCtx: TypeContext) (symbolRegistry: SymbolRegistry) : Map<string, string> =
    analysis.PrunedAsts
    |> Map.map (fun filePath ast ->
        let initialState = createInitialState typeCtx symbolRegistry
        
        // Set current module context
        let modulePath = analysis.Unit.SourceFiles.[filePath].ModulePath
        let stateWithModule = 
            { initialState with CurrentModule = modulePath }
        
        let (_, finalState) = runBuilder (generateMLIR ast) stateWithModule
        finalState.Output.ToString())

/// Helper to merge multiple MLIR modules into a single module
let mergeMLIRModules (modules: Map<string, string>) (mainFile: string) : string =
    let sb = System.Text.StringBuilder()
    
    // Add module header
    sb.AppendLine("module @FireflyProgram {") |> ignore
    
    // Add main file first
    match Map.tryFind mainFile modules with
    | Some mainModule -> 
        sb.AppendLine("  // Main module") |> ignore
        sb.AppendLine(mainModule) |> ignore
    | None -> ()
    
    // Add other modules
    modules 
    |> Map.toList
    |> List.filter (fun (path, _) -> path <> mainFile)
    |> List.iter (fun (path, moduleContent) ->
        sb.AppendLine(sprintf "  // Module from %s" (Path.GetFileName(path))) |> ignore
        sb.AppendLine(moduleContent) |> ignore)
    
    sb.AppendLine("}") |> ignore
    sb.ToString()