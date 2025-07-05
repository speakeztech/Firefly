module Dabbit.Analysis.Patterns

open System
open System.Text.RegularExpressions

/// Represents a symbol reference found in AST
type SymbolReference = {
    Name: string
    NodeType: string // "Call", "Value", "NewUnionCase", etc.
    Location: string // Line context for debugging
}

/// AST pattern extraction for F# Compiler Service output
module PatternExtractor =
    
    /// Extract function calls from Call expressions
    /// Pattern: Call (None, val functionName, [], [], [], [...])
    let extractFunctionCalls (astContent: string) =
        let pattern = @"Call\s*\(\s*None,\s*val\s+(\w+),"
        let regex = Regex(pattern, RegexOptions.Multiline)
        
        regex.Matches(astContent)
        |> Seq.cast<Match>
        |> Seq.map (fun m -> {
            Name = m.Groups.[1].Value
            NodeType = "Call"
            Location = m.Value
        })
        |> Seq.toList
    
    /// Extract member calls  
    /// Pattern: Call (Some Value val obj, member memberName, [type], [], [], [...])
    let extractMemberCalls (astContent: string) =
        let pattern = @"Call\s*\(\s*Some\s+Value\s+val\s+(\w+),\s*member\s+(\w+),"
        let regex = Regex(pattern, RegexOptions.Multiline)
        
        regex.Matches(astContent)
        |> Seq.cast<Match>
        |> Seq.map (fun m -> {
            Name = m.Groups.[2].Value // The member name
            NodeType = "MemberCall"
            Location = m.Value
        })
        |> Seq.toList
    
    /// Extract value references
    /// Pattern: Value val valueName
    let extractValueReferences (astContent: string) =
        let pattern = @"Value\s+val\s+(\w+)"
        let regex = Regex(pattern, RegexOptions.Multiline)
        
        regex.Matches(astContent)
        |> Seq.cast<Match>
        |> Seq.map (fun m -> {
            Name = m.Groups.[1].Value
            NodeType = "Value"
            Location = m.Value
        })
        |> Seq.toList
    
    /// Extract function definitions
    /// Pattern: MemberOrFunctionOrValue (val functionName, [[params]], body)
    let extractFunctionDefinitions (astContent: string) =
        let pattern = @"MemberOrFunctionOrValue\s*\(\s*val\s+(\w+),"
        let regex = Regex(pattern, RegexOptions.Multiline)
        
        regex.Matches(astContent)
        |> Seq.cast<Match>
        |> Seq.map (fun m -> m.Groups.[1].Value)
        |> Set.ofSeq
    
    /// Extract union case constructors
    /// Pattern: NewUnionCase (type TypeName, Constructor, [...])
    let extractUnionCaseUsage (astContent: string) =
        let pattern = @"NewUnionCase\s*\(\s*type\s+[\w\.]+<?[\w\.]*>?,\s*(\w+),"
        let regex = Regex(pattern, RegexOptions.Multiline)
        
        regex.Matches(astContent)
        |> Seq.cast<Match>
        |> Seq.map (fun m -> {
            Name = m.Groups.[1].Value
            NodeType = "UnionCase"
            Location = m.Value
        })
        |> Seq.toList
    
    /// Extract all symbol references from AST
    let extractAllReferences (astContent: string) =
        [
            extractFunctionCalls astContent
            extractMemberCalls astContent
            extractValueReferences astContent
            extractUnionCaseUsage astContent
        ]
        |> List.concat
        |> List.distinctBy (fun ref -> ref.Name)

/// AST node analysis for structural understanding
module StructureAnalyzer =
    
    /// Extract entity (module/namespace) structure
    let extractEntityNames (astContent: string) =
        let pattern = @"Entity\s*\(\s*(\w+),"
        let regex = Regex(pattern, RegexOptions.Multiline)
        
        regex.Matches(astContent)
        |> Seq.cast<Match>
        |> Seq.map (fun m -> m.Groups.[1].Value)
        |> Seq.toList
    
    /// Check if AST contains entry point functions
    let hasEntryPoints (astContent: string) =
        let entryPointPatterns = [
            @"val\s+main\s*,"
            @"val\s+hello\s*,"
            @"val\s+\w*Main\s*,"
        ]
        
        entryPointPatterns
        |> List.exists (fun pattern ->
            let regex = Regex(pattern, RegexOptions.IgnoreCase)
            regex.IsMatch(astContent))
    
    /// Extract type definitions
    let extractTypeDefinitions (astContent: string) =
        let pattern = @"type\s+(\w+)"
        let regex = Regex(pattern, RegexOptions.Multiline)
        
        regex.Matches(astContent)
        |> Seq.cast<Match>
        |> Seq.map (fun m -> m.Groups.[1].Value)
        |> Set.ofSeq

/// AST pruning utilities
module PruningUtilities =
    
    /// Extract a function block from AST content
    let extractFunctionBlock (astContent: string) (functionName: string) =
        let lines = astContent.Split('\n')
        let pattern = sprintf @"MemberOrFunctionOrValue\s*\(\s*val\s+%s," (Regex.Escape(functionName))
        let regex = Regex(pattern)
        
        let rec findFunctionStart lineIndex =
            if lineIndex >= lines.Length then None
            elif regex.IsMatch(lines.[lineIndex]) then Some lineIndex
            else findFunctionStart (lineIndex + 1)
        
        match findFunctionStart 0 with
        | None -> None
        | Some startIndex ->
            let rec findFunctionEnd currentIndex depth =
                if currentIndex >= lines.Length then currentIndex - 1
                else
                    let line = lines.[currentIndex]
                    let openCount = line |> Seq.filter (fun c -> c = '(') |> Seq.length
                    let closeCount = line |> Seq.filter (fun c -> c = ')') |> Seq.length
                    let newDepth = depth + openCount - closeCount
                    
                    if newDepth = 0 && currentIndex > startIndex then
                        currentIndex
                    else
                        findFunctionEnd (currentIndex + 1) newDepth
            
            let endIndex = findFunctionEnd startIndex 0
            let functionLines = lines.[startIndex..endIndex]
            Some (String.Join("\n", functionLines))
    
    /// Check if a line contains a function definition
    let isFunctionDefinition (line: string) =
        let pattern = @"MemberOrFunctionOrValue\s*\(\s*val\s+\w+,"
        let regex = Regex(pattern)
        regex.IsMatch(line)
    
    /// Extract function name from definition line
    let extractFunctionNameFromLine (line: string) =
        let pattern = @"MemberOrFunctionOrValue\s*\(\s*val\s+(\w+),"
        let regex = Regex(pattern)
        let match_ = regex.Match(line)
        
        if match_.Success then
            Some match_.Groups.[1].Value
        else
            None