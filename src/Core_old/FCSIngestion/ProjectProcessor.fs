module Core.FCSIngestion.ProjectProcessor

open System
open FSharp.Compiler.CodeAnalysis

/// Result of processing a project through FCS
type ProcessedProject = {
    /// The raw FCS project results
    CheckResults: FSharpCheckProjectResults
    
    /// All symbol uses with locations
    SymbolUses: FSharpSymbolUse[]
    
    /// Elapsed time for processing
    ProcessingTime: TimeSpan
}

/// Errors from project processing
type ProcessingError = {
    CriticalErrors: FSharp.Compiler.Diagnostics.FSharpDiagnostic[]
    AllDiagnostics: FSharp.Compiler.Diagnostics.FSharpDiagnostic[]
}

/// Process a project using FCS ParseAndCheckProject
let processProject (projectPath: string) (projectOptions: FSharpProjectOptions) (checker: FSharpChecker) =
    async {
       
        // Process the project
        let startTime = DateTime.UtcNow
        
        printfn "[ProjectProcessor] Processing project: %s" projectPath
        printfn "[ProjectProcessor] Source files: %d" projectOptions.SourceFiles.Length
        
        let! projectResults = checker.ParseAndCheckProject(projectOptions)
        
        let processingTime = DateTime.UtcNow - startTime
        printfn "[ProjectProcessor] ParseAndCheckProject completed in %.2f seconds" processingTime.TotalSeconds
        
        // Check for errors
        let criticalErrors = 
            projectResults.Diagnostics 
            |> Array.filter (fun d -> d.Severity = FSharp.Compiler.Diagnostics.FSharpDiagnosticSeverity.Error)
        
        printfn "[ProjectProcessor] Diagnostics:"
        printfn "  Total: %d" projectResults.Diagnostics.Length
        printfn "  Errors: %d" criticalErrors.Length
        printfn "  Warnings: %d" (projectResults.Diagnostics.Length - criticalErrors.Length)
        
        if projectResults.HasCriticalErrors then
            printfn "[ProjectProcessor] ❌ Critical errors found"
            criticalErrors |> Array.truncate 10 |> Array.iter (fun err ->
                printfn "  [ERROR] %s:%d:%d - %s" 
                    err.FileName err.StartLine err.StartColumn err.Message)
            
            return Error {
                CriticalErrors = criticalErrors
                AllDiagnostics = projectResults.Diagnostics
            }
        else
            printfn "[ProjectProcessor] ✅ No critical errors"
            
            // Get all symbol uses
            printfn "[ProjectProcessor] Extracting symbol uses..."
            let allSymbolUses = projectResults.GetAllUsesOfAllSymbols()
            printfn "[ProjectProcessor] Found %d symbol uses" allSymbolUses.Length
            
            return Ok {
                CheckResults = projectResults
                SymbolUses = allSymbolUses
                ProcessingTime = processingTime
            }
    }