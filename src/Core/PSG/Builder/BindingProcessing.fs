/// Binding processing for PSG construction
module Core.PSG.Construction.BindingProcessing

open FSharp.Compiler.Text
open FSharp.Compiler.Syntax
open FSharp.Compiler.Symbols
open Core.PSG.Types
open Core.PSG.Construction.Types
open Core.PSG.Construction.SymbolCorrelation
open Core.PSG.Construction.PatternProcessing
open Core.PSG.Construction.ExpressionProcessing

/// Process binding with explicit use flag
let rec processBindingWithUseFlag binding parentId fileName (context: BuildContext) graph isUse =
    match binding with
    | SynBinding(accessibility, kind, isInline, isMutable, attributes, xmlDoc, valData, pat, returnInfo, expr, range, seqPoint, trivia) ->
        // Get symbol from the PATTERN, not from loose range matching on the binding
        let symbol = extractSymbolFromPattern pat fileName context.CorrelationContext

        // Check for EntryPoint attribute
        let hasEntryPointAttr =
            attributes
            |> List.exists (fun attrList ->
                attrList.Attributes
                |> List.exists (fun attr ->
                    let attrName =
                        match attr.TypeName with
                        | SynLongIdent(idents, _, _) ->
                            idents |> List.map (fun id -> id.idText) |> String.concat "."
                    attrName = "EntryPoint" || attrName = "EntryPointAttribute"))

        // Detect main function pattern
        let isMainFunc =
            match pat with
            | SynPat.LongIdent(SynLongIdent(idents, _, _), _, _, _, _, _) ->
                idents |> List.exists (fun id -> id.idText = "main")
            | SynPat.Named(SynIdent(ident, _), _, _, _) ->
                ident.idText = "main"
            | _ -> false

        // Create binding node with enhanced detection
        let syntaxKind =
            if hasEntryPointAttr then "Binding:EntryPoint"
            elif isMainFunc then "Binding:Main"
            elif isUse then "Binding:Use"
            else "Binding"

        let bindingNode = createNode syntaxKind range fileName symbol parentId

        let graph' = { graph with Nodes = Map.add bindingNode.Id.Value bindingNode graph.Nodes }
        let graph'' = addChildToParent bindingNode.Id parentId graph'

        let graph''' =
            match symbol with
            | Some sym ->
                let updatedSymbolTable = Map.add sym.DisplayName sym graph''.SymbolTable

                // Add to entry points if this is an entry point
                let updatedEntryPoints =
                    if hasEntryPointAttr || isMainFunc then
                        printfn "[BUILDER] Found entry point during construction: %s" sym.FullName
                        bindingNode.Id :: graph''.EntryPoints
                    else
                        graph''.EntryPoints

                { graph'' with
                    SymbolTable = updatedSymbolTable
                    EntryPoints = updatedEntryPoints }
            | None ->
                printfn "[BUILDER] Warning: No symbol correlation for binding at %s" (range.ToString())
                graph''

        let graph'''' = processPattern pat (Some bindingNode.Id) fileName context graph'''
        processExpression expr (Some bindingNode.Id) fileName context graph''''

/// Regular binding processing (for non-LetOrUse contexts)
let processBinding binding parentId fileName context graph =
    processBindingWithUseFlag binding parentId fileName context graph false

/// Initialize the circular reference in ExpressionProcessing
let ensureInitialized () =
    setBindingProcessor processBindingWithUseFlag

// Force initialization on module load
do ensureInitialized ()
