module Dabbit.Transformations.StackAllocation

open FSharp.Compiler.Syntax
open FSharp.Compiler.Text
open Dabbit.CodeGeneration.TypeMapping

/// Stack allocation analysis
type AllocInfo = {
    Size: int option
    IsFixed: bool
    Source: string
}

/// Analyze and transform allocations to stack
module StackTransform =
    /// Identify allocation patterns
    let (|HeapAlloc|_|) = function
        | SynExpr.App(_, _, SynExpr.Ident ident, size, _) 
            when ident.idText = "Array.zeroCreate" || ident.idText = "Array.create" ->
            Some (size, ident.idText)
        | SynExpr.ArrayOrList(true, elements, _) ->
            Some (SynExpr.Const(SynConst.Int32 elements.Length, range.Zero), "array literal")
        | _ -> None
    
    /// Extract constant size
    let (|ConstSize|_|) = function
        | SynExpr.Const(SynConst.Int32 n, _) -> Some n
        | _ -> None
    
    /// Transform allocations to stack allocations
    let rec transform expr =
        match expr with
        | HeapAlloc(ConstSize size, source) when size <= 1024 ->
            // Transform to stack allocation
            let range = expr.Range
            SynExpr.App(
                ExprAtomicFlag.NonAtomic, false,
                SynExpr.App(
                    ExprAtomicFlag.NonAtomic, false,
                    SynExpr.Ident(Ident("NativePtr.stackalloc", range)),
                    SynExpr.Const(SynConst.Int32 size, range),
                    range),
                SynExpr.Ident(Ident("Span", range)),
                range)
        
        | SynExpr.App(flag, infix, func, arg, range) ->
            SynExpr.App(flag, infix, transform func, transform arg, range)
        
        | SynExpr.LetOrUse(isRec, isUse, bindings, body, range, trivia) ->
            let bindings' = bindings |> List.map (transformBinding)
            SynExpr.LetOrUse(isRec, isUse, bindings', transform body, range, trivia)
        
        | SynExpr.IfThenElse(cond, thenExpr, elseOpt, sp, isFromTry, range, trivia) ->
            SynExpr.IfThenElse(transform cond, transform thenExpr, 
                             Option.map transform elseOpt, sp, isFromTry, range, trivia)
        
        | SynExpr.Match(sp, matchExpr, clauses, range, trivia) ->
            let clauses' = clauses |> List.map (fun (SynMatchClause(pat, when', result, r, sp, tr)) ->
                SynMatchClause(pat, Option.map transform when', transform result, r, sp, tr))
            SynExpr.Match(sp, transform matchExpr, clauses', range, trivia)
        
        | SynExpr.Sequential(sp, isTrueSeq, e1, e2, range, trivia) ->
            SynExpr.Sequential(sp, isTrueSeq, transform e1, transform e2, range, trivia)
        
        | _ -> expr
    
    and transformBinding (SynBinding(access, kind, isInline, isMut, attrs, xmlDoc, 
                                   valData, headPat, retInfo, expr, range, sp, trivia)) =
        SynBinding(access, kind, isInline, isMut, attrs, xmlDoc, valData,
                   headPat, retInfo, transform expr, range, sp, trivia)

    /// Transform a module's declarations
    let rec transformModuleDecl = function
        | SynModuleDecl.Let(isRec, bindings, range) ->
            let bindings' = bindings |> List.map transformBinding
            SynModuleDecl.Let(isRec, bindings', range)
        | SynModuleDecl.Expr(expr, range) ->
            SynModuleDecl.Expr(transform expr, range)
        | SynModuleDecl.NestedModule(componentInfo, isRec, decls, isCont, range, trivia) ->
            let decls' = decls |> List.map transformModuleDecl
            SynModuleDecl.NestedModule(componentInfo, isRec, decls', isCont, range, trivia)
        | other -> other

    /// Transform an entire module
    let transformModule (SynModuleOrNamespace(lid, isRec, kind, decls, xml, attrs, access, range, trivia)) =
        let decls' = decls |> List.map transformModuleDecl
        SynModuleOrNamespace(lid, isRec, kind, decls', xml, attrs, access, range, trivia)

/// Verify stack safety
module StackSafety =
    /// Check if expression only uses stack allocations
    let rec verify expr =
        match expr with
        | SynExpr.App(_, _, SynExpr.Ident ident, _, _) ->
            match ident.idText with
            | "Array.zeroCreate" | "Array.create" | "List.init" -> 
                Error "Heap allocation detected"
            | "NativePtr.stackalloc" | "Span" -> 
                Ok ()
            | _ -> Ok ()
        
        | SynExpr.ArrayOrList(_, _, _) ->
            Error "Array/List literal creates heap allocation"
        
        | SynExpr.App(_, _, func, arg, _) ->
            Result.bind (fun _ -> verify arg) (verify func)
        
        | SynExpr.LetOrUse(_, _, bindings, body, _, _) ->
            let bindResults = bindings |> List.map verifyBinding
            match List.tryFind Result.isError bindResults with
            | Some err -> err
            | None -> verify body
        
        | SynExpr.Sequential(_, _, e1, e2, _, _) ->
            Result.bind (fun _ -> verify e2) (verify e1)
        
        | _ -> Ok ()
    
    and verifyBinding (SynBinding(_, _, _, _, _, _, _, _, _, expr, _, _, _)) =
        verify expr
        
module Application =
    /// Apply stack allocation transformation to the entire AST
    let applyStackAllocation (ast: ParsedInput) (typeCtx: TypeContext) (reachableSymbols: Set<string>) : ParsedInput =
        // Helper to check if a symbol is reachable
        let isReachable qualifiedName =
            Set.contains qualifiedName reachableSymbols
        
        match ast with
        | ParsedInput.ImplFile(ParsedImplFileInput(fileName, isScript, qualName, pragmas, hashDirectives, modules, isLastCompiled, isExe, trivia)) ->
            // Transform all modules
            let transformedModules = 
                modules |> List.map (fun moduleOrNamespace ->
                    match moduleOrNamespace with
                    | SynModuleOrNamespace(lid, isRec, kind, decls, xml, attrs, access, range, trivia) ->
                        // Build the module path for reachability checking
                        let modulePath = lid |> List.map (fun id -> id.idText) |> String.concat "."
                        
                        // Transform declarations, considering reachability
                        let transformedDecls = 
                            decls |> List.map (fun decl ->
                                match decl with
                                | SynModuleDecl.Let(isRec, bindings, range) ->
                                    let transformedBindings = 
                                        bindings |> List.map (fun binding ->
                                            match binding with
                                            | SynBinding(access, kind, isInline, isMut, attrs, xmlDoc, 
                                                        valData, headPat, retInfo, expr, bindingRange, sp, trivia) ->
                                                // Extract the binding name for reachability check
                                                let bindingName = 
                                                    match headPat with
                                                    | SynPat.LongIdent(longDotId, _, _, _, _, _) ->
                                                        let ids = match longDotId with
                                                                    | SynLongIdent(ids, _, _) -> ids
                                                        ids |> List.map (fun id -> id.idText) |> String.concat "."
                                                    | SynPat.Named(SynIdent(id, _), _, _, _) -> id.idText
                                                    | _ -> ""
                                                
                                                let fullName = 
                                                    if modulePath = "" then bindingName
                                                    else sprintf "%s.%s" modulePath bindingName
                                                
                                                // Only transform if the symbol is reachable
                                                if isReachable fullName then
                                                    let transformedExpr = StackTransform.transform expr
                                                    SynBinding(access, kind, isInline, isMut, attrs, xmlDoc,
                                                            valData, headPat, retInfo, transformedExpr, 
                                                            bindingRange, sp, trivia)
                                                else
                                                    binding)
                                    SynModuleDecl.Let(isRec, transformedBindings, range)
                                
                                | SynModuleDecl.Expr(expr, range) ->
                                    // Transform standalone expressions
                                    SynModuleDecl.Expr(StackTransform.transform expr, range)
                                
                                | SynModuleDecl.NestedModule(componentInfo, isRec, nestedDecls, isCont, range, trivia) ->
                                    // Recursively handle nested modules
                                    StackTransform.transformModuleDecl decl
                                
                                | other -> other)
                        
                        SynModuleOrNamespace(lid, isRec, kind, transformedDecls, xml, attrs, access, range, trivia))
            
            ParsedInput.ImplFile(ParsedImplFileInput(fileName, isScript, qualName, pragmas, hashDirectives, 
                                                    transformedModules, isLastCompiled, isExe, trivia))
        
        | ParsedInput.SigFile _ ->
            // Signature files don't need stack allocation transformation
            ast