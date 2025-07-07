# 🚨 CRITICAL FCS 43.9.300 API MEMO FOR CLAUDE

## Hey Claude, STOP making these mistakes with FCS 43.9.300:

### 1. RANGE TYPE AND IMPORTS
- **WRONG**: `Range.range` 
- **RIGHT**: `range` (from `open FSharp.Compiler.Text`)
- **ALWAYS IMPORT**: `open FSharp.Compiler.Text` when using range types
- **TYPE**: Range is just `range`, not `Range.range`

### 2. RANGE PROPERTIES - YOU KEEP SCREWING THIS UP
- **WRONG**: `range.StartLine`, `range.StartColumn`, `range.EndLine`, `range.EndColumn`
- **RIGHT**: `range.Start.Line`, `range.Start.Column`, `range.End.Line`, `range.End.Column`
- **STRUCTURE**: Range has `.Start` and `.End` properties, each with `.Line` and `.Column`
- **FILENAME**: `range.FileName` (this one is correct)

### 3. CHECK RESULTS HIERARCHY - STOP CONFUSING THESE
- **FSharpCheckProjectResults**: Has `.AssemblyContents.ImplementationFiles` ✅
- **FSharpCheckFileResults**: Does NOT have AssemblyContents ❌
- **ProcessedProject.CheckResults**: Is FSharpCheckProjectResults (confirmed in codebase)
- **WHEN TO USE WHAT**: 
  - Project-level analysis (functions, modules) → FSharpCheckProjectResults
  - File-level analysis (individual file symbols) → FSharpCheckFileResults

### 4. IMPLEMENTATION FILE PROPERTIES
- **QualifiedName**: Already a string, NOT a sequence
- **WRONG**: `implFile.QualifiedName |> String.concat "."`
- **RIGHT**: `implFile.QualifiedName` (it's already concatenated)

### 5. PATH OPERATIONS TYPE ANNOTATIONS
- **ISSUE**: `Path.GetFileName()` needs explicit type when range.FileName might be ambiguous
- **SOLUTION**: Cast or use explicit string type: `Path.GetFileName(f.Range.FileName : string)`
- **OR**: Just avoid Path.GetFileName in diagnostic output to prevent type issues

### 6. FSHARP EXPR TRAVERSAL PATTERNS
- **Match**: Use proper FSharpExpr cases like:
  - `FSharpExpr.Call(objExprOpt, memberFunc, _, _, args)`
  - `FSharpExpr.Application(funcExpr, _, args)`
  - `FSharpExpr.Let(binding, body)`
  - `FSharpExpr.Sequential(expr1, expr2)`
- **Don't assume**: Complex expression hierarchies - keep traversal simple

### 7. SYMBOL ATTRIBUTE CHECKING
- **EntryPoint**: `attr.AttributeType.DisplayName = "EntryPoint"` works
- **Alternative**: Check for both DisplayName and BasicQualifiedName patterns
- **Main function**: `value.LogicalName = "main" && value.IsModuleValueOrMember`

### 8. DIAGNOSTIC OUTPUT - KEEP IT SIMPLE
- **Avoid**: Complex range formatting until you know the API works
- **Start with**: Just function names and basic info
- **Add complexity**: Only after core functionality works

### 9. PROJECT VS FILE SCOPE CONFUSION
- **Project Analysis**: Extract functions across all files → use FSharpCheckProjectResults
- **File Analysis**: Analyze individual file symbols → use FSharpCheckFileResults  
- **Symbol Uses**: Available on both but with different scopes

### 10. COMMON COMPILATION FIXES
- **When you see**: "Object does not define field StartLine" → You used wrong range API
- **When you see**: "Object does not define field Start" → range might not be the type you think
- **When you see**: "string not compatible with string seq" → You're trying to concat something already concatenated
- **When you see**: "AssemblyContents not found" → You used FSharpCheckFileResults instead of FSharpCheckProjectResults

## 🎯 GOLDEN RULES FOR FCS 43.9.300:
1. **ALWAYS** import `FSharp.Compiler.Text` for range
2. **ALWAYS** use `range.Start.Line` not `range.StartLine` 
3. **ALWAYS** use FSharpCheckProjectResults for multi-file analysis
4. **ALWAYS** check if properties are already processed (like QualifiedName)
5. **NEVER** assume Range.range - it's just `range`
6. **TEST** basic compilation before adding fancy formatting

## 🔥 EMERGENCY PATTERN FOR DEBUGGING:
```fsharp
// When in doubt, use this minimal pattern:
let testFunction (checkResults: FSharpCheckProjectResults) =
    checkResults.AssemblyContents.ImplementationFiles
    |> Seq.iter (fun implFile ->
        printfn "Module: %s" implFile.QualifiedName
        // Add complexity incrementally from here
    )
```

**STOP WASTING TIME ON THESE BASIC API ERRORS. GET THE FOUNDATION RIGHT FIRST.**