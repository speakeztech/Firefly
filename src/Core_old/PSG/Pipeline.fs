module Core.PSG.Pipeline

open System.IO
open FSharp.Compiler.CodeAnalysis
open Core.PSG.Types
open Core.PSG.Builder
open Core.PSG.IntermediateWriter

/// Options for PSG construction
type PSGOptions = {
    GenerateIntermediates: bool
    IntermediatesDir: string option
    ValidateGraph: bool
    VerboseOutput: bool
}

/// Default PSG options
let defaultPSGOptions = {
    GenerateIntermediates = true
    IntermediatesDir = None
    ValidateGraph = true
    VerboseOutput = false
}

/// Simple progress reporter function
type ProgressReporter = string -> unit

/// Create a PSG from project check results
let createPSG 
    (checker: FSharpChecker) 
    (projectOptions: FSharpProjectOptions)
    (options: PSGOptions)
    (progress: ProgressReporter) : Async<FcsResult<ProgramSemanticGraph>> =
    
    async {
        try
            // Parse all project files
            progress "Parsing source files..."
            // Modify this line in Pipeline.fs:
            let! parseResults = 
                // Get parsing options from project options
                let parsingOptions, _ = checker.GetParsingOptionsFromProjectOptions projectOptions
                
                projectOptions.SourceFiles
                |> Array.map (fun file -> 
                    // Read file content and create source text
                    let content = System.IO.File.ReadAllText(file)
                    let sourceText = FSharp.Compiler.Text.SourceText.ofString content
                    
                    // Use the correct overload with the right parameters
                    checker.ParseFile(file, sourceText, parsingOptions))
                |> Async.Parallel

            // Type check the project
            progress "Type checking project..."
            let! checkResults = checker.ParseAndCheckProject(projectOptions)
            
            // Report any errors from type checking
            if checkResults.HasCriticalErrors then
                let errors = 
                    checkResults.Diagnostics
                    |> Seq.filter (fun diag -> diag.Severity = FSharp.Compiler.Diagnostics.FSharpDiagnosticSeverity.Error)
                    |> Seq.map (fun diag ->
                        {
                            Severity = Error
                            Code = "FCS" + diag.ErrorNumber.ToString()
                            Message = diag.Message
                            Location = Some diag.Range
                            RelatedLocations = []
                        })
                    |> Seq.toList
                
                return Failure errors
            else
                // Load source files content
                progress "Loading source files content..."
                let sourceFiles =
                    projectOptions.SourceFiles
                    |> Array.choose (fun file ->
                        try
                            Some (file, File.ReadAllText(file))
                        with _ ->
                            None)
                    |> Map.ofArray
                
                // Build PSG
                progress "Building Program Semantic Graph..."
                let psgResult = buildPSG parseResults checkResults sourceFiles
                
                // Generate intermediates if requested
                match psgResult with
                | Success psg ->
                    if options.GenerateIntermediates then
                        let intermediatesDir = 
                            match options.IntermediatesDir with
                            | Some dir -> dir
                            | None -> 
                                let defaultDir = Path.Combine(Path.GetDirectoryName(projectOptions.SourceFiles.[0]), "intermediates")
                                if options.VerboseOutput then
                                    progress (sprintf "Using default intermediates directory: %s" defaultDir)
                                defaultDir
                        
                        progress "Generating intermediate assets..."
                        writeIntermediates intermediatesDir parseResults checkResults psg
                    
                    return Success psg
                | Failure errors -> return Failure errors
        with ex ->
            return Failure [{
                Severity = DiagnosticSeverity.Error
                Code = "PSG000"
                Message = sprintf "PSG creation failed: %s" ex.Message
                Location = None
                RelatedLocations = []
            }]
    }

/// Validate a PSG for correctness
let validatePSG (psg: ProgramSemanticGraph) : FcsResult<ProgramSemanticGraph> =
    let diagnostics = ResizeArray<DiagnosticMessage>()
    
    // Check for orphaned nodes
    let allNodeIds = 
        Set.unionMany [
            Set.ofSeq (psg.ModuleNodes |> Map.keys)
            Set.ofSeq (psg.TypeNodes |> Map.keys)
            Set.ofSeq (psg.ValueNodes |> Map.keys)
            Set.ofSeq (psg.ExpressionNodes |> Map.keys)
            Set.ofSeq (psg.PatternNodes |> Map.keys)
        ]
    
    // Check all child references are valid
    let orphanedChildRefs =
        let allChildRefs = 
            seq {
                yield! psg.ModuleNodes |> Map.values |> Seq.collect (fun node -> node.Children)
                yield! psg.TypeNodes |> Map.values |> Seq.collect (fun node -> node.Children)
                yield! psg.ValueNodes |> Map.values |> Seq.collect (fun node -> node.Children)
                yield! psg.ExpressionNodes |> Map.values |> Seq.collect (fun node -> node.Children)
                yield! psg.PatternNodes |> Map.values |> Seq.collect (fun node -> node.Children)
            }
        
        allChildRefs
        |> Seq.filter (fun childId -> not (Set.contains childId allNodeIds))
        |> Seq.toList
    
    if not (List.isEmpty orphanedChildRefs) then
        diagnostics.Add({
            Severity = DiagnosticSeverity.Error
            Code = "PSG101"
            Message = sprintf "Found %d orphaned child references" orphanedChildRefs.Length
            Location = None
            RelatedLocations = []
        })
    
    // Check for entry points that don't exist in the node map
    let invalidEntryPoints =
        psg.EntryPoints
        |> List.filter (fun entryPoint -> not (Set.contains entryPoint allNodeIds))
    
    if not (List.isEmpty invalidEntryPoints) then
        diagnostics.Add({
            Severity = DiagnosticSeverity.Error
            Code = "PSG102"
            Message = sprintf "Found %d invalid entry points" invalidEntryPoints.Length
            Location = None
            RelatedLocations = []
        })
    
    // Return validation result
    if diagnostics.Count = 0 then
        Success psg
    else
        Failure (List.ofSeq diagnostics)