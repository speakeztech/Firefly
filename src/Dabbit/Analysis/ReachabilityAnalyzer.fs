module Dabbit.Analysis.ReachabilityAnalyzer

open System
open System.IO
open System.Collections.Generic

/// Represents a symbol reference in the AST
type SymbolReference = {
    FullName: string
    SourceModule: string
    NodeType: string // "Call", "Value", "NewUnionCase", etc.
}

/// AST node visitor for extracting symbol references
type ASTVisitor() =
    let references = ResizeArray<SymbolReference>()
    
    /// Extract symbol references from AST content
    member this.VisitASTContent(content: string, sourceModule: string) =
        // Parse different types of references from the AST text
        this.ExtractCallReferences(content, sourceModule)
        this.ExtractValueReferences(content, sourceModule)
        this.ExtractTypeReferences(content, sourceModule)
        
    /// Extract Call expressions like "Call (None, val functionName, ...)"
    member private this.ExtractCallReferences(content: string, sourceModule: string) =
        let lines = content.Split('\n')
        for line in lines do
            if line.Contains("Call") && line.Contains("val ") then
                // Parse patterns like: Call (None, val writeBytes, [], [], [], ...)
                let valStart = line.IndexOf("val ") + 4
                if valStart > 3 then
                    let remaining = line.Substring(valStart)
                    let commaIndex = remaining.IndexOf(',')
                    if commaIndex > 0 then
                        let functionName = remaining.Substring(0, commaIndex).Trim()
                        references.Add({
                            FullName = functionName
                            SourceModule = sourceModule
                            NodeType = "Call"
                        })
    
    /// Extract Value references like "Value val functionName"
    member private this.ExtractValueReferences(content: string, sourceModule: string) =
        let lines = content.Split('\n')
        for line in lines do
            if line.Contains("Value val ") then
                let valStart = line.IndexOf("Value val ") + 10
                if valStart > 9 then
                    let remaining = line.Substring(valStart)
                    let endChars = [|';'; ')'; ']'; ','; ' '|]
                    let endIndex = remaining.IndexOfAny(endChars)
                    if endIndex > 0 then
                        let valueName = remaining.Substring(0, endIndex).Trim()
                        references.Add({
                            FullName = valueName
                            SourceModule = sourceModule
                            NodeType = "Value"
                        })
    
    /// Extract type references like "NewUnionCase", "NewObject", etc.
    member private this.ExtractTypeReferences(content: string, sourceModule: string) =
        let lines = content.Split('\n')
        for line in lines do
            // Extract NewUnionCase references
            if line.Contains("NewUnionCase") then
                // Parse: NewUnionCase (type SomeType, Constructor, [...])
                // TODO: Implement type extraction
                ()
            
            // Extract NewObject references  
            if line.Contains("NewObject") then
                // Parse: NewObject (member .ctor, [type], [...])
                // TODO: Implement constructor extraction
                ()
    
    member this.GetReferences() = references.ToArray()
    member this.Clear() = references.Clear()

/// Module dependency tracker
type ModuleDependencyTracker() =
    let dependencies = Dictionary<string, HashSet<string>>()
    
    member this.AddDependency(fromModule: string, toSymbol: string) =
        if not (dependencies.ContainsKey(fromModule)) then
            dependencies.[fromModule] <- HashSet<string>()
        dependencies.[fromModule].Add(toSymbol) |> ignore
    
    member this.GetDependencies(moduleName: string) =
        match dependencies.TryGetValue(moduleName) with
        | true, deps -> deps |> Set.ofSeq
        | false, _ -> Set.empty
    
    member this.AllDependencies = 
        dependencies 
        |> Seq.map (fun kvp -> kvp.Key, kvp.Value |> Set.ofSeq)
        |> Map.ofSeq

/// Main reachability analyzer for AST files
type ASTReachabilityAnalyzer(astFiles: (string * string) list) =
    let visitor = ASTVisitor()
    let depTracker = ModuleDependencyTracker()
    let allSymbols = Dictionary<string, string>() // symbol -> defining module
    
    /// Parse all AST files and build symbol tables
    member this.BuildSymbolTables() =
        for (moduleName, astContent) in astFiles do
            // Extract all defined symbols in this module
            this.ExtractDefinedSymbols(moduleName, astContent)
            
            // Extract all symbol references  
            visitor.Clear()
            visitor.VisitASTContent(astContent, moduleName)
            
            // Build dependency graph
            for reference in visitor.GetReferences() do
                depTracker.AddDependency(moduleName, reference.FullName)
    
    /// Extract symbols defined in a module
    member private this.ExtractDefinedSymbols(moduleName: string, astContent: string) =
        let lines = astContent.Split('\n')
        for line in lines do
            // Look for MemberOrFunctionOrValue definitions
            if line.Contains("MemberOrFunctionOrValue") then
                let valStart = line.IndexOf("(val ") + 5
                if valStart > 4 then
                    let remaining = line.Substring(valStart)
                    let commaIndex = remaining.IndexOf(',')
                    if commaIndex > 0 then
                        let symbolName = remaining.Substring(0, commaIndex).Trim()
                        let fullName = sprintf "%s.%s" moduleName symbolName
                        allSymbols.[symbolName] <- moduleName
                        allSymbols.[fullName] <- moduleName
    
    /// Perform reachability analysis starting from entry points
    member this.AnalyzeReachability(entryPoints: string list) =
        this.BuildSymbolTables()
        
        let reachable = HashSet<string>()
        let worklist = Queue<string>()
        
        // Add entry points to worklist
        for entry in entryPoints do
            worklist.Enqueue(entry)
            reachable.Add(entry) |> ignore
        
        // Process worklist until empty
        while worklist.Count > 0 do
            let current = worklist.Dequeue()
            
            // Find which module defines this symbol
            match allSymbols.TryGetValue(current) with
            | true, definingModule ->
                // Get all dependencies of this module
                let deps = depTracker.GetDependencies(definingModule)
                for dep in deps do
                    if not (reachable.Contains(dep)) then
                        reachable.Add(dep) |> ignore
                        worklist.Enqueue(dep)
            | false, _ -> 
                // Symbol not found - might be external or resolved differently
                printfn $"Warning: Symbol not found: {current}"
        
        reachable |> Set.ofSeq
    
    /// Generate pruned AST for a specific module
    member this.GeneratePrunedAST(moduleName: string, reachableSymbols: Set<string>) =
        // Find the original AST for this module
        match astFiles |> List.tryFind (fun (name, _) -> name = moduleName) with
        | Some (_, originalAST) ->
            // TODO: Implement actual AST pruning logic
            // For now, just return original AST if any symbols are reachable
            let moduleSymbols = 
                allSymbols 
                |> Seq.filter (fun kvp -> kvp.Value = moduleName)
                |> Seq.map (fun kvp -> kvp.Key)
                |> Set.ofSeq
            
            let hasReachableSymbols = 
                Set.intersect moduleSymbols reachableSymbols
                |> Set.isEmpty
                |> not
            
            if hasReachableSymbols then Some originalAST else None
        | None -> None
    
    /// Generate final AST with all reachable dependencies inlined
    member this.GenerateFinalAST(mainModule: string, reachableSymbols: Set<string>) =
        // TODO: Implement final AST generation
        // This should include all reachable symbols from all modules
        // in a single computation graph for MLIR conversion
        
        let reachableModules = 
            reachableSymbols
            |> Set.toSeq
            |> Seq.choose (fun symbol -> 
                match allSymbols.TryGetValue(symbol) with
                | true, moduleName -> Some moduleName
                | false, _ -> None)
            |> Set.ofSeq
        
        // Combine ASTs from all reachable modules
        let combinedAST = 
            astFiles
            |> List.filter (fun (moduleName, _) -> Set.contains moduleName reachableModules)
            |> List.map snd
            |> String.concat "\n\n"
        
        combinedAST

/// Helper functions for file I/O
module ASTFileProcessor =
    
    /// Load AST files from directory with numeric prefixes
    let loadASTFiles(directory: string) =
        Directory.GetFiles(directory, "*.initial.ast")
        |> Array.map (fun filePath ->
            let fileName = Path.GetFileNameWithoutExtension(filePath)
            let moduleName = fileName.Replace(".initial", "")
            let content = File.ReadAllText(filePath)
            (moduleName, content))
        |> Array.toList
    
    /// Write pruned AST files
    let writePrunedASTs(directory: string, prunedASTs: (string * string) list) =
        for (moduleName, astContent) in prunedASTs do
            let fileName = sprintf "%s.pruned.ast" moduleName
            let filePath = Path.Combine(directory, fileName)
            File.WriteAllText(filePath, astContent)
    
    /// Write final AST file
    let writeFinalAST(directory: string, mainModule: string, finalAST: string) =
        let fileName = sprintf "%s.final.ast" mainModule
        let filePath = Path.Combine(directory, fileName)
        File.WriteAllText(filePath, finalAST)

/// Example usage
module Example =
    let runReachabilityAnalysis() =
        // Load all initial AST files
        let astFiles = ASTFileProcessor.loadASTFiles("./build/intermediates")
        
        // Create analyzer
        let analyzer = ASTReachabilityAnalyzer(astFiles)
        
        // Analyze reachability starting from main
        let reachableSymbols = analyzer.AnalyzeReachability(["main"; "hello"])
        
        printfn $"Found {Set.count reachableSymbols} reachable symbols"
        
        // Generate pruned ASTs for each module
        let prunedASTs = 
            astFiles
            |> List.map (fun (moduleName, _) -> moduleName)
            |> List.choose (fun moduleName ->
                match analyzer.GeneratePrunedAST(moduleName, reachableSymbols) with
                | Some prunedAST -> Some (moduleName, prunedAST)
                | None -> None)
        
        // Generate final AST
        let finalAST = analyzer.GenerateFinalAST("06_01_HelloWorldDirect", reachableSymbols)
        
        // Write output files
        ASTFileProcessor.writePrunedASTs("./build/intermediates", prunedASTs)
        ASTFileProcessor.writeFinalAST("./build/intermediates", "06_01_HelloWorldDirect", finalAST)
        
        printfn "Reachability analysis complete!"
        printfn $"Generated {List.length prunedASTs} pruned AST files"
        printfn "Generated final AST file"