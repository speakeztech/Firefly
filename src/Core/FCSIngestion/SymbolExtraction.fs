module Core.FCSIngestion.SymbolExtraction

open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Symbols

/// Extract symbols defined in a specific file
let extractDefinedSymbols (checkResults: FSharpCheckFileResults) (fileName: string) =
    let assemblySig = checkResults.PartialAssemblySignature
    let shortName = System.IO.Path.GetFileName(fileName)
    
    {|
        FileName = fileName
        Modules = 
            assemblySig.Entities 
            |> Seq.filter (fun e -> e.IsFSharpModule && e.DeclarationLocation.FileName.EndsWith(shortName))
            |> Seq.map (fun e -> e.FullName)
            |> Seq.toArray
        Types = 
            assemblySig.Entities
            |> Seq.filter (fun e -> not e.IsFSharpModule && e.DeclarationLocation.FileName.EndsWith(shortName))
            |> Seq.map (fun e -> {| Name = e.FullName; Kind = e.DisplayName |})
            |> Seq.toArray
        Functions = 
            assemblySig.Entities
            |> Seq.collect (fun e -> e.MembersFunctionsAndValues)
            |> Seq.filter (fun f -> f.DeclarationLocation.FileName.EndsWith(shortName))
            |> Seq.map (fun f -> {| Name = f.FullName; Signature = f.FullType.Format(FSharpDisplayContext.Empty) |})
            |> Seq.toArray
    |}

/// Extract all symbols visible at this point (cumulative)
let extractCumulativeSymbols (checkResults: FSharpCheckFileResults) =
    let assemblySig = checkResults.PartialAssemblySignature
    
    {|
        TotalModules = assemblySig.Entities |> Seq.filter (fun e -> e.IsFSharpModule) |> Seq.length
        TotalTypes = assemblySig.Entities |> Seq.filter (fun e -> not e.IsFSharpModule) |> Seq.length
        TotalFunctions = assemblySig.Entities |> Seq.collect (fun e -> e.MembersFunctionsAndValues) |> Seq.length
        ModuleNames = assemblySig.Entities |> Seq.filter (fun e -> e.IsFSharpModule) |> Seq.map (fun e -> e.FullName) |> Seq.toArray
    |}