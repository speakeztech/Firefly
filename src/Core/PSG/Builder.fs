module Core.PSG.Builder

open FSharp.Compiler.Syntax
open FSharp.Compiler.Text
open FSharp.Compiler.Symbols
open FSharp.Compiler.CodeAnalysis
open Core.PSG.Types
open Core.PSG.XmlDocParser

/// Initialize an empty PSG
let createEmptyPSG() : ProgramSemanticGraph = {
    SourceASTs = Map.empty
    ModuleNodes = Map.empty
    TypeNodes = Map.empty
    ValueNodes = Map.empty
    ExpressionNodes = Map.empty
    PatternNodes = Map.empty
    SymbolTable = Map.empty
    RangeToSymbol = Map.empty
    SymbolToNodes = Map.empty
    TypeDefinitions = Map.empty
    DependencyGraph = Map.empty
    EntryPoints = []
    AlloyReferences = Set.empty
    SourceFiles = Map.empty
}

/// Extract source text for a range if available
let getSourceText (sourceFiles: Map<string, string>) (range: range) : string option =
    match Map.tryFind range.FileName sourceFiles with
    | Some source ->
        try
            let startPos = range.Start.Line - 1 // 0-based line indexing
            let lines = source.Split('\n')
            
            if startPos >= 0 && startPos < lines.Length then
                let line = lines.[startPos]
                let startCol = range.Start.Column
                let endCol = if range.Start.Line = range.End.Line then range.End.Column else line.Length
                
                if startCol <= endCol && endCol <= line.Length then
                    Some (line.Substring(startCol, endCol - startCol))
                else
                    None
            else
                None
        with _ ->
            None
    | None -> None

/// Create a source location record
let createSourceLocation (sourceFiles: Map<string, string>) (range: range) : SourceLocation =
    {
        Range = range
        OriginalSourceText = getSourceText sourceFiles range
    }

/// Add a symbol to the PSG
let addSymbol (symbol: FSharpSymbol) (psg: ProgramSemanticGraph) : ProgramSemanticGraph =
    let symbolKey = sprintf "%s_%d" (symbol.ToString()) (symbol.GetHashCode())
    { psg with SymbolTable = Map.add symbolKey symbol psg.SymbolTable }

/// Add a range-to-symbol mapping
let addRangeSymbolMapping (range: range) (symbol: FSharpSymbol) (psg: ProgramSemanticGraph) : ProgramSemanticGraph =
    let rangeKey = RangeKey.FromRange(range)
    { psg with RangeToSymbol = Map.add rangeKey symbol psg.RangeToSymbol }

/// Add a symbol-to-node mapping
let addSymbolNodeMapping (symbol: FSharpSymbol) (nodeId: NodeId) (psg: ProgramSemanticGraph) : ProgramSemanticGraph =
    let symbolKey = sprintf "%s_%d" (symbol.ToString()) (symbol.GetHashCode())
    let existingNodes = 
        match Map.tryFind symbolKey psg.SymbolToNodes with
        | Some nodes -> nodes
        | None -> Set.empty
    
    let updatedNodes = Set.add nodeId existingNodes
    { psg with SymbolToNodes = Map.add symbolKey updatedNodes psg.SymbolToNodes }

/// Find a symbol for a given range
let tryFindSymbolForRange (range: range) (symbolUses: FSharpSymbolUse[]) : FSharpSymbol option =
    symbolUses
    |> Array.tryFind (fun symbolUse -> 
        // Use just Range instead of RangeAlternate
        let symbolRange = symbolUse.Range
        symbolRange.Start.Line = range.Start.Line &&
        symbolRange.Start.Column = range.Start.Column &&
        symbolRange.End.Line = range.End.Line &&
        symbolRange.End.Column = range.End.Column)
    |> Option.map (fun symbolUse -> symbolUse.Symbol)

/// Get MLIR metadata for a symbol
let getMetadataForSymbol (symbol: FSharpSymbol) : MLIRMetadata option =
    match extractXmlDocFromMemberOrFunction symbol with
    | Some xmlDoc ->
        match parseMLIRMetadata xmlDoc with
        | Valid metadata -> Some metadata
        | _ -> None
    | None -> None

/// Process a syntax node and enrich it with symbol information
let rec processNode 
    (node: 'TNode) 
    (nodeType: string)
    (range: range) 
    (parentId: NodeId option)
    (symbolUses: FSharpSymbolUse[])
    (sourceFiles: Map<string, string>) : EnrichedNode<'TNode> =
    
    // Create node ID
    let nodeId = NodeId.FromSyntaxNode(nodeType, range)
    
    // Try to find symbol for this range
    let symbol = tryFindSymbolForRange range symbolUses
    
    // Get metadata if symbol is available
    let metadata = 
        match symbol with
        | Some sym -> getMetadataForSymbol sym
        | None -> None
    
    // Create source location
    let sourceLocation = createSourceLocation sourceFiles range
    
    // Create enriched node
    {
        Syntax = node
        Symbol = symbol
        Metadata = metadata
        SourceLocation = sourceLocation
        Id = nodeId
        ParentId = parentId
        Children = [] // Will be populated by specific node processors
    }

/// Process module or namespace declaration
let rec processModuleOrNamespace 
    (modOrNs: SynModuleOrNamespace) 
    (parentId: NodeId option)
    (symbolUses: FSharpSymbolUse[])
    (sourceFiles: Map<string, string>)
    (psg: ProgramSemanticGraph) : ProgramSemanticGraph * EnrichedNode<SynModuleOrNamespace> =
    
    // Create enriched node for module
    let moduleNode = processNode modOrNs "module" modOrNs.Range parentId symbolUses sourceFiles
    
    // Extract declarations through pattern matching
    let (SynModuleOrNamespace(_, _, _, declarations, _, _, _, _, _)) = modOrNs
    
    // Process module declarations
    let psgWithDecls, childIds = 
        ((psg, []), declarations)
        ||> List.fold (fun (currentPsg, ids) decl ->
            let updatedPsg, childId = processDeclaration decl (Some moduleNode.Id) symbolUses sourceFiles currentPsg
            (updatedPsg, childId :: ids))
    
    // Update module node with children
    let updatedModuleNode = { moduleNode with Children = childIds }
    
    // Add to PSG
    let updatedPsg = { psgWithDecls with ModuleNodes = Map.add updatedModuleNode.Id updatedModuleNode psgWithDecls.ModuleNodes }
    
    // Add symbol mapping if available
    let finalPsg =
        match moduleNode.Symbol with
        | Some symbol -> 
            updatedPsg 
            |> addSymbol symbol
            |> addRangeSymbolMapping moduleNode.SourceLocation.Range symbol
            |> addSymbolNodeMapping symbol moduleNode.Id
        | None -> updatedPsg
    
    finalPsg, updatedModuleNode

/// Process a declaration within a module
and processDeclaration 
    (decl: SynModuleDecl) 
    (parentId: NodeId option)
    (symbolUses: FSharpSymbolUse[])
    (sourceFiles: Map<string, string>)
    (psg: ProgramSemanticGraph) : ProgramSemanticGraph * NodeId =
    
    match decl with
    | SynModuleDecl.Let(_, bindings, range) -> // Add trivia parameter
        // Process each binding
        let psgWithBindings, childIds = 
            ((psg, []), bindings)
            ||> List.fold (fun (currentPsg, ids) binding ->
                let updatedPsg, node = processBinding binding parentId symbolUses sourceFiles currentPsg
                (updatedPsg, node.Id :: ids))
        
        // Create a node for the let declaration group
        let letNodeId = NodeId.FromSyntaxNode("let_decl", range)
        psgWithBindings, letNodeId
        
    | SynModuleDecl.Types(typeDefs, range) -> // Add trivia parameter
        // Process each type definition
        let psgWithTypes, childIds = 
            ((psg, []), typeDefs)
            ||> List.fold (fun (currentPsg, ids) typeDef ->
                let updatedPsg, node = processTypeDefinition typeDef parentId symbolUses sourceFiles currentPsg
                (updatedPsg, node.Id :: ids))
        
        // Return the last type ID (simplification - in real code, create a container node)
        match childIds with
        | id :: _ -> psgWithTypes, id
        | [] -> psgWithTypes, NodeId.FromSyntaxNode("empty_types", Range.Zero)
        
    | SynModuleDecl.NestedModule(componentInfo, isRec, decls, _, range, _) -> // Add trivia parameter
        // Create node for nested module
        let moduleNodeId = NodeId.FromSyntaxNode("nested_module", range)
        
        // Process declarations in nested module
        let psgWithDecls, _ = 
            ((psg, []), decls)
            ||> List.fold (fun (currentPsg, ids) nestedDecl ->
                let updatedPsg, childId = processDeclaration nestedDecl (Some moduleNodeId) symbolUses sourceFiles currentPsg
                (updatedPsg, childId :: ids))
        
        psgWithDecls, moduleNodeId
        
    | _ ->
        // Handle other declaration types as needed
        let declNodeId = NodeId.FromSyntaxNode("other_decl", decl.Range)
        psg, declNodeId

/// Process a binding (value or function)
and processBinding 
    (binding: SynBinding) 
    (parentId: NodeId option)
    (symbolUses: FSharpSymbolUse[])
    (sourceFiles: Map<string, string>)
    (psg: ProgramSemanticGraph) : ProgramSemanticGraph * EnrichedNode<SynBinding> =
    
    // Use RangeOfBindingWithRhs to get the binding range
    let bindingRange = binding.RangeOfBindingWithRhs 
    
    // Create enriched node for binding
    let bindingNode = processNode binding "binding" bindingRange parentId symbolUses sourceFiles
    
    // Extract the expression directly from the binding through pattern matching
    let (SynBinding(_, _, _, _, _, _, _, _, _, expr, _, _, _)) = binding
    
    // Process the expression
    let psgWithExpr, exprNode = processExpression expr (Some bindingNode.Id) symbolUses sourceFiles psg
    
    // Update binding node with expression as child
    let updatedBindingNode = { bindingNode with Children = [exprNode.Id] }
    
    // Add to PSG
    let updatedPsg = { psgWithExpr with ValueNodes = Map.add updatedBindingNode.Id updatedBindingNode psgWithExpr.ValueNodes }
    
    // Add symbol mapping if available
    let finalPsg =
        match bindingNode.Symbol with
        | Some symbol -> 
            updatedPsg 
            |> addSymbol symbol
            |> addRangeSymbolMapping bindingNode.SourceLocation.Range symbol
            |> addSymbolNodeMapping symbol bindingNode.Id
        | None -> updatedPsg
    
    finalPsg, updatedBindingNode

/// Process an expression
and processExpression 
    (expr: SynExpr) 
    (parentId: NodeId option)
    (symbolUses: FSharpSymbolUse[])
    (sourceFiles: Map<string, string>)
    (psg: ProgramSemanticGraph) : ProgramSemanticGraph * EnrichedNode<SynExpr> =
    
    // Create enriched node for expression
    let exprNode = processNode expr "expr" expr.Range parentId symbolUses sourceFiles
    
    // Process sub-expressions based on expression type
    let psgWithSubExprs, childIds = 
        match expr with
        | SynExpr.App(_, _, funcExpr, argExpr, _) ->
            // Process function and argument expressions
            let psgWithFunc, funcNode = processExpression funcExpr (Some exprNode.Id) symbolUses sourceFiles psg
            let psgWithArg, argNode = processExpression argExpr (Some exprNode.Id) symbolUses sourceFiles psgWithFunc
            psgWithArg, [funcNode.Id; argNode.Id]
            
        | SynExpr.LetOrUse(isRecursive, isUse, bindings, bodyExpr, range, _) -> // Add trivia parameter
            // Process bindings and body
            let psgWithBindings, bindingIds = 
                ((psg, []), bindings)
                ||> List.fold (fun (currentPsg, ids) binding ->
                    let updatedPsg, node = processBinding binding (Some exprNode.Id) symbolUses sourceFiles currentPsg
                    (updatedPsg, node.Id :: ids))
            
            let psgWithBody, bodyNode = processExpression bodyExpr (Some exprNode.Id) symbolUses sourceFiles psgWithBindings
            psgWithBody, bodyNode.Id :: bindingIds
            
        | SynExpr.Lambda(_, _, _, bodyExpr, _, _, _) -> // Add missing parameters
            // We'll simplify lambda handling for now - just process the body
            let psgWithBody, bodyNode = processExpression bodyExpr (Some exprNode.Id) symbolUses sourceFiles psg
            psgWithBody, [bodyNode.Id]
            
        | _ ->
            // Handle other expression types
            psg, []
    
    // Update expression node with children
    let updatedExprNode = { exprNode with Children = childIds }
    
    // Add to PSG
    let updatedPsg = { psgWithSubExprs with ExpressionNodes = Map.add updatedExprNode.Id updatedExprNode psgWithSubExprs.ExpressionNodes }
    
    // Add symbol mapping if available
    let finalPsg =
        match exprNode.Symbol with
        | Some symbol -> 
            updatedPsg 
            |> addSymbol symbol
            |> addRangeSymbolMapping exprNode.SourceLocation.Range symbol
            |> addSymbolNodeMapping symbol exprNode.Id
        | None -> updatedPsg
    
    finalPsg, updatedExprNode

/// Process a type definition
and processTypeDefinition 
    (typeDef: SynTypeDefn) 
    (parentId: NodeId option)
    (symbolUses: FSharpSymbolUse[])
    (sourceFiles: Map<string, string>)
    (psg: ProgramSemanticGraph) : ProgramSemanticGraph * EnrichedNode<SynTypeDefn> =
    
    // Create enriched node for type definition
    let typeDefNode = processNode typeDef "type_def" typeDef.Range parentId symbolUses sourceFiles
    
    // Add to PSG
    let updatedPsg = { psg with TypeNodes = Map.add typeDefNode.Id typeDefNode psg.TypeNodes }
    
    // Add symbol mapping if available
    let finalPsg =
        match typeDefNode.Symbol with
        | Some symbol -> 
            updatedPsg 
            |> addSymbol symbol
            |> addRangeSymbolMapping typeDefNode.SourceLocation.Range symbol
            |> addSymbolNodeMapping symbol typeDefNode.Id
        | None -> updatedPsg
    
    finalPsg, typeDefNode

/// Build a complete PSG from parsed and checked results
let buildPSG 
    (parseResults: FSharpParseFileResults[]) 
    (checkResults: FSharpCheckProjectResults)
    (sourceFiles: Map<string, string>) : FcsResult<ProgramSemanticGraph> =
    
    try
        // Get all symbol uses
        let symbolUses = checkResults.GetAllUsesOfAllSymbols()
        
        // Create empty PSG
        let psg = createEmptyPSG()
        
        // Add source files to PSG
        let psgWithSources = { psg with SourceFiles = sourceFiles }
        
        // Add parsed inputs to PSG
        let psgWithInputs = 
            (psgWithSources, parseResults)
            ||> Array.fold (fun psg parseResult ->
                let parsedInput = parseResult.ParseTree
                { psg with SourceASTs = Map.add parseResult.FileName parsedInput psg.SourceASTs })
        
        // Process each parsed input
        let finalPsg = 
            (psgWithInputs, parseResults)
            ||> Array.fold (fun psg parseResult ->
                let parsedInput = parseResult.ParseTree
                match parsedInput with
                | ParsedInput.ImplFile implFile ->
                    let (ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) = implFile
                    // Process each module or namespace
                    (psg, modules)
                    ||> List.fold (fun currentPsg modOrNs ->
                        let updatedPsg, _ = processModuleOrNamespace modOrNs None symbolUses sourceFiles currentPsg
                        updatedPsg)
                | ParsedInput.SigFile _ -> 
                    // Handle signature files if needed
                    psg)
        
        // STEP 1: Collect all potential entry points
        let potentialEntryPoints = 
            checkResults.AssemblyContents.ImplementationFiles
            |> Seq.collect (fun implFile ->
                implFile.Declarations
                |> Seq.collect (fun decl ->
                    match decl with
                    | FSharpImplementationFileDeclaration.Entity (entity, subDecls) -> 
                        // Look inside entities (modules, namespaces)
                        subDecls |> Seq.collect (function
                            | FSharpImplementationFileDeclaration.MemberOrFunctionOrValue(mfv, _, _) ->
                                let hasEntryPointAttr = 
                                    mfv.Attributes 
                                    |> Seq.exists (fun attr -> 
                                        attr.AttributeType.BasicQualifiedName = "EntryPointAttribute" ||
                                        attr.AttributeType.BasicQualifiedName = "System.EntryPointAttribute")
                                
                                let isMainFunction = 
                                    mfv.LogicalName = "main" && 
                                    mfv.FullType.IsFunctionType
                                
                                if hasEntryPointAttr || isMainFunction then
                                    // Store the full path for lookup
                                    let fullPath = sprintf "%s.%s.%s" 
                                                        implFile.QualifiedName 
                                                        entity.DisplayName 
                                                        mfv.DisplayName
                                    
                                    printfn "Found entry point: %s in %s (attr: %b, main: %b)" 
                                        mfv.DisplayName fullPath hasEntryPointAttr isMainFunction
                                    
                                    [(fullPath, mfv)]
                                else []
                            | _ -> [])
                    | FSharpImplementationFileDeclaration.MemberOrFunctionOrValue(mfv, _, _) ->
                        // Top-level functions (rare but possible)
                        let hasEntryPointAttr = 
                            mfv.Attributes 
                            |> Seq.exists (fun attr -> 
                                attr.AttributeType.BasicQualifiedName = "EntryPointAttribute" ||
                                attr.AttributeType.BasicQualifiedName = "System.EntryPointAttribute")
                        
                        let isMainFunction = 
                            mfv.LogicalName = "main" && 
                            mfv.FullType.IsFunctionType
                        
                        if hasEntryPointAttr || isMainFunction then
                            let fullPath = sprintf "%s.%s" 
                                                implFile.QualifiedName 
                                                mfv.DisplayName
                            
                            printfn "Found entry point: %s in %s (attr: %b, main: %b)" 
                                mfv.DisplayName fullPath hasEntryPointAttr isMainFunction
                            
                            [(fullPath, mfv)]
                        else []
                    | _ -> []
                )
            )
            |> Seq.toList
        
        // STEP 2: Map entry points to nodes
        let entryPointNodes = 
            potentialEntryPoints
            |> List.choose (fun (path, symbol) ->
                // Try to find a node corresponding to this symbol
                let symbolKey = symbol.FullName
                printfn "Looking for symbol key: %s" symbolKey
                
                // Try to find in SymbolToNodes map
                match Map.tryFind symbolKey finalPsg.SymbolToNodes with
                | Some nodeIds when not (Set.isEmpty nodeIds) -> 
                    let nodeId = Set.minElement nodeIds
                    printfn "Found node for entry point: %s -> %s" symbolKey nodeId.Value
                    Some nodeId
                | _ -> 
                    // Try by display name as fallback
                    let displayKey = symbol.DisplayName
                    match Map.tryFind displayKey finalPsg.SymbolToNodes with
                    | Some nodeIds when not (Set.isEmpty nodeIds) -> 
                        let nodeId = Set.minElement nodeIds
                        printfn "Found node by display name: %s -> %s" displayKey nodeId.Value
                        Some nodeId
                    | _ ->
                        // Last resort - scan through for a matching pattern
                        let matchingNodes =
                            finalPsg.ValueNodes
                            |> Map.toSeq
                            |> Seq.choose (fun (nodeId, node) ->
                                match node.Symbol with
                                | Some sym when sym.DisplayName = symbol.DisplayName -> Some nodeId
                                | _ -> None)
                            |> Seq.tryHead
                        
                        match matchingNodes with
                        | Some nodeId -> 
                            printfn "Found node by scan: %s" nodeId.Value
                            Some nodeId
                        | None -> 
                            printfn "WARNING: Failed to map entry point to node: %s" path
                            printfn "Symbols in map: %d, Nodes: %d" 
                                (Map.count finalPsg.SymbolTable) 
                                (Map.count finalPsg.ValueNodes)
                            None
            )

        // Print summary of entry point detection
        printfn "DEBUG: Found %d entry points out of %d functions" 
            entryPointNodes.Length potentialEntryPoints.Length
        
        // STEP 3: Update PSG with entry points
        let psgWithEntryPoints = { finalPsg with EntryPoints = entryPointNodes }
        
        // Collect Alloy references for zero-allocation analysis
        let alloyRefs = 
            symbolUses
            |> Seq.choose (fun symbolUse ->
                let symbol = symbolUse.Symbol
                match symbol with
                | :? FSharpEntity as entity when entity.FullName.StartsWith("Alloy.") ->
                    Some entity.FullName
                | :? FSharpMemberOrFunctionOrValue as mfv when 
                    mfv.FullName.StartsWith("Alloy.") || 
                    (mfv.DeclaringEntity.IsSome && mfv.DeclaringEntity.Value.FullName.StartsWith("Alloy.")) ->
                    Some mfv.FullName
                | _ -> None)
            |> Set.ofSeq
        
        // Final PSG with Alloy references
        let completePsg = { psgWithEntryPoints with AlloyReferences = alloyRefs }
        
        Success completePsg
    with ex ->
        Failure [{
            Severity = DiagnosticSeverity.Error
            Code = "FCS001"  // FCS-specific error code
            Message = sprintf "Failed to build PSG: %s" ex.Message
            Location = None
            RelatedLocations = []
        }]