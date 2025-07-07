module Core.AST.Extraction

open FSharp.Compiler.Symbols
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Text

type TypedFunction = {
    Symbol: FSharpMemberOrFunctionOrValue
    FullName: string
    Range: range
    Body: FSharpExpr
    Module: string
    IsEntryPoint: bool
}


let extractFunctions (checkResults: FSharpCheckProjectResults) : TypedFunction[] =
    
    // Track all type NAMES defined in this project (using strings instead of FSharpEntity)
    let projectDefinedTypeNames (checkResults: FSharpCheckProjectResults) =
        try
            checkResults.AssemblyContents.ImplementationFiles
            |> Seq.collect (fun file ->
                file.Declarations
                |> Seq.collect (fun decl ->
                    let rec collectTypeNames (declaration: FSharpImplementationFileDeclaration) =
                        seq {
                            match declaration with
                            | FSharpImplementationFileDeclaration.Entity (entity, subDecls) ->
                                // Try to safely get the FullName, skipping types without one
                                try 
                                    // Only include non-compiler types (exclude FSharp.Core internals)
                                    if not (entity.CompiledName.StartsWith("_")) && 
                                       not (entity.CompiledName.Contains("@")) then
                                        yield entity.FullName  // Get the name string instead of the entity
                                with _ -> 
                                    // Skip types that throw when accessing FullName
                                    ()
                                    
                                // Recursively collect from nested declarations
                                yield! subDecls |> Seq.collect collectTypeNames
                            | _ -> 
                                // Skip other declaration types
                                ()
                        }
                    decl |> collectTypeNames))
            |> Set.ofSeq  // Now we're creating a Set<string> which is valid
        with ex ->
            printfn "Warning: Error collecting types: %s" ex.Message
            Set.empty
    
    // Get defined type names with error handling
    let definedTypeNames = projectDefinedTypeNames checkResults
    
    // Extract functions with proper source tracking and error handling
    try
        checkResults.AssemblyContents.ImplementationFiles
        |> Seq.collect (fun implFile ->
            let moduleName = implFile.QualifiedName
            let sourceFile = implFile.FileName
            
            let rec processDeclarations (decls: FSharpImplementationFileDeclaration list) =
                decls |> List.choose (fun decl ->
                    try
                        match decl with
                        | FSharpImplementationFileDeclaration.MemberOrFunctionOrValue(value, _, body) 
                            when value.IsFunction ->
                            /// Detect if a function is an entry point using standard F# rules
                            let isEntryPoint (value: FSharpMemberOrFunctionOrValue) (_moduleName: string) =
                                try
                                    // Check for EntryPoint attribute (primary rule in F#)
                                    let hasEntryPointAttribute = 
                                        value.Attributes 
                                        |> Seq.exists (fun attr -> 
                                            try attr.AttributeType.DisplayName = "EntryPoint"
                                            with _ -> false)
                                    
                                    // Check for main function with appropriate signature
                                    let isMainFunction = 
                                        (value.LogicalName = "main" || value.DisplayName = "main") &&
                                        value.IsModuleValueOrMember
                                    
                                    // Debug output
                                    if isMainFunction || hasEntryPointAttribute then
                                        printfn "Found potential entry point: %s (attribute: %b, main function: %b)" 
                                            value.LogicalName hasEntryPointAttribute isMainFunction
                                    
                                    hasEntryPointAttribute || isMainFunction
                                with ex ->
                                    printfn "Exception in isEntryPoint for %s: %s" value.LogicalName ex.Message
                                    false
                            
                            // Create TypedFunction with safe property access
                            try
                                Some {
                                    Symbol = value
                                    FullName = value.FullName
                                    Range = body.Range
                                    Body = body
                                    Module = moduleName
                                    IsEntryPoint = isEntryPoint value moduleName
                                }
                            with ex ->
                                printfn "Warning: Skipping function due to error: %s" ex.Message
                                None
                                
                        | FSharpImplementationFileDeclaration.Entity(_, subDecls) ->
                            // Process nested declarations
                            let nestedFunctions = processDeclarations subDecls
                            if nestedFunctions.IsEmpty then None
                            else nestedFunctions |> List.tryHead
                            
                        | _ -> None
                    with ex ->
                        printfn "Warning: Error processing declaration: %s" ex.Message
                        None)
            
            try
                processDeclarations implFile.Declarations
            with ex ->
                printfn "Warning: Error processing file %s: %s" implFile.FileName ex.Message
                [])
        |> Array.ofSeq
    with ex ->
        printfn "Error extracting functions: %s" ex.Message
        printfn "Stack trace: %s" ex.StackTrace
        [||]  // Return empty array if extraction fails

// Add this after your extractFunctions function
let debugEntryPoints (functions: TypedFunction[]) =
    let entryPoints = functions |> Array.filter (fun f -> f.IsEntryPoint)
    printfn "DEBUG: Found %d entry points out of %d functions" entryPoints.Length functions.Length
    entryPoints |> Array.iter (fun f -> 
        printfn "DEBUG: Entry point: %s (from %s)" f.FullName f.Module)