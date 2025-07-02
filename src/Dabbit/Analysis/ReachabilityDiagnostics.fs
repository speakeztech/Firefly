module Dabbit.Analysis.ReachabilityDiagnostics

open System
open System.IO
open FSharp.Compiler.Syntax
open FSharp.Compiler.Text

/// Resolution attempt result
type ResolutionResult =
    | ResolvedTo of qualifiedName: string
    | ResolvedLocal of bindingKind: ScopeManager.BindingKind
    | Failed of reason: ResolutionFailureReason

and ResolutionFailureReason =
    | NotFound
    | AmbiguousReference of candidates: string list
    | MemberNotFound of typeName: string * memberName: string
    | ModuleNotOpen of moduleName: string

/// Record types for diagnostic events
type SymbolLookupInfo = {
    Name: string
    Module: string list
    OpenModules: string list list
    Result: ResolutionResult
    Location: range option
}

type ReachabilityMarkInfo = {
    Symbol: string
    MarkedFrom: string
    Reason: string
}

type ScopeEntryInfo = {
    ScopeKind: ScopeManager.ScopeKind
    LocalBindings: string list
}

type TypeResolutionInfo = {
    Expression: string
    ResolvedType: string option
    Location: range option
}

type DependencyReason =
    | DirectCall
    | TypeReference
    | MemberAccess
    | PatternMatch
    | ModuleOpen
    | Initialization

type PruningDecision =
    | Keep
    | Remove

type DependencyEdgeInfo = {
    From: string
    To: string
    Reason: DependencyReason
    Location: range option
}

type PruningDecisionInfo = {
    Symbol: string
    Decision: PruningDecision
    Reason: string
}

/// Diagnostic event during reachability analysis
type DiagnosticEvent =
    | SymbolLookupAttempt of SymbolLookupInfo
    | DependencyEdgeAdded of DependencyEdgeInfo
    | ReachabilityMarked of ReachabilityMarkInfo
    | ScopeEntered of ScopeEntryInfo
    | ScopeExited of scopeKind: ScopeManager.ScopeKind
    | TypeResolution of TypeResolutionInfo
    | PruningDecision of PruningDecisionInfo
/// Immutable diagnostics state
type DiagnosticsState = {
    Events: DiagnosticEvent list
    CurrentModule: string list
}

/// Create empty diagnostics state
let empty = {
    Events = []
    CurrentModule = []
}

/// Record a diagnostic event
let recordEvent (event: DiagnosticEvent) (state: DiagnosticsState) =
    { state with Events = event :: state.Events }

/// Set current module context
let setCurrentModule (modulePath: string list) (state: DiagnosticsState) =
    { state with CurrentModule = modulePath }

/// Log a symbol lookup attempt
let logSymbolLookup (name: string) (openModules: string list list) (result: ResolutionResult) (location: range option) (state: DiagnosticsState) =
    let event = SymbolLookupAttempt {
        Name = name
        Module = state.CurrentModule
        OpenModules = openModules
        Result = result
        Location = location
    }
    recordEvent event state

/// Log a dependency edge
let logDependencyEdge (from: string) (to': string) (reason: DependencyReason) (location: range option) (state: DiagnosticsState) =
    let event = DependencyEdgeAdded {
        From = from
        To = to'
        Reason = reason
        Location = location
    }
    recordEvent event state

/// Log reachability marking
let logReachabilityMark (symbol: string) (markedFrom: string) (reason: string) (state: DiagnosticsState) =
    let event = ReachabilityMarked {
        Symbol = symbol
        MarkedFrom = markedFrom
        Reason = reason
    }
    recordEvent event state

/// Log scope entry
let logScopeEntry (scopeKind: ScopeManager.ScopeKind) (bindings: string list) (state: DiagnosticsState) =
    let event = ScopeEntered {
        ScopeKind = scopeKind
        LocalBindings = bindings
    }
    recordEvent event state

/// Log scope exit
let logScopeExit (scopeKind: ScopeManager.ScopeKind) (state: DiagnosticsState) =
    let event = ScopeExited scopeKind
    recordEvent event state

/// Log type resolution
let logTypeResolution (expr: string) (resolvedType: string option) (location: range option) (state: DiagnosticsState) =
    let event = TypeResolution {
        Expression = expr
        ResolvedType = resolvedType
        Location = location
    }
    recordEvent event state

/// Log pruning decision
let logPruningDecision (symbol: string) (decision: PruningDecision) (reason: string) (state: DiagnosticsState) =
    let event = PruningDecision {
        Symbol = symbol
        Decision = decision
        Reason = reason
    }
    recordEvent event state

/// Get all events in chronological order
let getEvents (state: DiagnosticsState) = 
    List.rev state.Events

/// Generate dependency graph in DOT format
let generateDependencyGraph (state: DiagnosticsState) =
    let edges = 
        state.Events
        |> List.choose (function
            | DependencyEdgeAdded info -> Some (info.From, info.To, info.Reason)
            | _ -> None)
    
    let nodes = 
        edges 
        |> List.collect (fun (from, to', _) -> [from; to'])
        |> List.distinct
    
    let dot = System.Text.StringBuilder()
    dot.AppendLine("digraph Dependencies {") |> ignore
    dot.AppendLine("  rankdir=LR;") |> ignore
    dot.AppendLine("  node [shape=box];") |> ignore
    
    // Add nodes
    nodes |> List.iter (fun node ->
        let label = node.Replace(".", "\\n")
        dot.AppendLine(sprintf "  \"%s\" [label=\"%s\"];" node label) |> ignore
    )
    
    // Add edges with labels
    edges |> List.iter (fun (from, to', reason) ->
        let label = 
            match reason with
            | DirectCall -> "calls"
            | TypeReference -> "references"
            | MemberAccess -> "accesses"
            | PatternMatch -> "matches"
            | ModuleOpen -> "opens"
            | Initialization -> "initializes"
        dot.AppendLine(sprintf "  \"%s\" -> \"%s\" [label=\"%s\"];" from to' label) |> ignore
    )
    
    dot.AppendLine("}") |> ignore
    dot.ToString()

/// Generate resolution failure report
let generateResolutionReport (state: DiagnosticsState) =
    let failures = 
        state.Events
        |> List.choose (function
            | SymbolLookupAttempt info ->
                match info.Result with
                | Failed reason -> Some (info.Name, info.Module, info.OpenModules, reason, info.Location)
                | _ -> None
            | _ -> None)
        |> List.groupBy (fun (name, _, _, _, _) -> name)
    
    let report = Text.StringBuilder()
    report.AppendLine("=== Resolution Failure Report ===") |> ignore
    report.AppendLine() |> ignore
    
    failures |> List.iter (fun (name, instances) ->
        report.AppendLine(sprintf "Symbol: %s" name) |> ignore
        report.AppendLine(sprintf "Failed %d times" (List.length instances)) |> ignore
        
        instances |> List.iter (fun (_, module', opens, reason, loc) ->
            report.AppendLine(sprintf "  Module: %s" (String.concat "." module')) |> ignore
            report.AppendLine(sprintf "  Open modules: %s" (opens |> List.map (String.concat ".") |> String.concat ", ")) |> ignore
            report.AppendLine(sprintf "  Reason: %A" reason) |> ignore
            match loc with
            | Some range -> 
                report.AppendLine(sprintf "  Location: %s (%d,%d)" range.FileName range.StartLine range.StartColumn) |> ignore
            | None -> ()
            report.AppendLine() |> ignore
        )
    )
    
    report.ToString()

/// Generate reachability statistics
let generateStatistics (state: DiagnosticsState) =
    let symbolsFound = 
        state.Events
        |> List.choose (function
            | ReachabilityMarked info -> Some info.Symbol
            | _ -> None)
        |> List.distinct
        |> List.length
    
    let resolutionAttempts = 
        state.Events
        |> List.filter (function SymbolLookupAttempt _ -> true | _ -> false)
        |> List.length
    
    let resolutionFailures = 
        state.Events
        |> List.filter (function 
            | SymbolLookupAttempt info -> 
                match info.Result with Failed _ -> true | _ -> false
            | _ -> false)
        |> List.length
    
    let pruningDecisions = 
        state.Events
        |> List.choose (function
            | PruningDecision info -> Some info.Decision
            | _ -> None)
    
    let kept = pruningDecisions |> List.filter ((=) Keep) |> List.length
    let removed = pruningDecisions |> List.filter ((=) Remove) |> List.length
    
    sprintf """=== Reachability Analysis Statistics ===
Symbols marked reachable: %d
Resolution attempts: %d
Resolution failures: %d (%.1f%%)
Pruning decisions:
  Kept: %d
  Removed: %d
  Elimination rate: %.1f%%
""" 
        symbolsFound 
        resolutionAttempts 
        resolutionFailures 
        (if resolutionAttempts > 0 then float resolutionFailures / float resolutionAttempts * 100.0 else 0.0)
        kept
        removed
        (if kept + removed > 0 then float removed / float (kept + removed) * 100.0 else 0.0)

/// Write diagnostics to file
let writeToFile (filePath: string) (state: DiagnosticsState) =
    use writer = new StreamWriter(filePath)
    
    // Write summary statistics
    writer.WriteLine(generateStatistics state)
    writer.WriteLine()
    
    // Write resolution failures
    writer.WriteLine(generateResolutionReport state)
    writer.WriteLine()
    
    // Write detailed event log
    writer.WriteLine("=== Detailed Event Log ===")
    writer.WriteLine()
    
    state.Events
    |> List.rev
    |> List.iter (fun event ->
        match event with
        | SymbolLookupAttempt info ->
            writer.WriteLine(sprintf "[LOOKUP] %s -> %A" info.Name info.Result)
        | DependencyEdgeAdded info ->
            writer.WriteLine(sprintf "[DEPENDENCY] %s -> %s (%A)" info.From info.To info.Reason)
        | ReachabilityMarked info ->
            writer.WriteLine(sprintf "[REACHABLE] %s (from %s: %s)" info.Symbol info.MarkedFrom info.Reason)
        | ScopeEntered info ->
            writer.WriteLine(sprintf "[SCOPE ENTER] %A with bindings: %s" info.ScopeKind (String.concat ", " info.LocalBindings))
        | ScopeExited scopeKind ->
            writer.WriteLine(sprintf "[SCOPE EXIT] %A" scopeKind)
        | TypeResolution info ->
            writer.WriteLine(sprintf "[TYPE] %s : %s" info.Expression (Option.defaultValue "unknown" info.ResolvedType))
        | PruningDecision info ->
            writer.WriteLine(sprintf "[PRUNE] %s -> %A (%s)" info.Symbol info.Decision info.Reason)
    )
    
    // Write dependency graph
    writer.WriteLine()
    writer.WriteLine("=== Dependency Graph (DOT format) ===")
    writer.WriteLine()
    writer.WriteLine(generateDependencyGraph state)

/// Thread-safe wrapper for diagnostics state
type DiagnosticsCollector() =
    let mutable state = empty
    let lockObj = obj()
    
    member _.RecordEvent(event) =
        lock lockObj (fun () ->
            state <- recordEvent event state
        )
    
    member _.SetCurrentModule(modulePath) =
        lock lockObj (fun () ->
            state <- setCurrentModule modulePath state
        )
    
    member _.LogSymbolLookup(name, openModules, result, location) =
        lock lockObj (fun () ->
            state <- logSymbolLookup name openModules result location state
        )
    
    member _.LogDependencyEdge(from, to', reason, location) =
        lock lockObj (fun () ->
            state <- logDependencyEdge from to' reason location state
        )
    
    member _.LogReachabilityMark(symbol, markedFrom, reason) =
        lock lockObj (fun () ->
            state <- logReachabilityMark symbol markedFrom reason state
        )
    
    member _.LogScopeEntry(scopeKind, bindings) =
        lock lockObj (fun () ->
            state <- logScopeEntry scopeKind bindings state
        )
    
    member _.LogScopeExit(scopeKind) =
        lock lockObj (fun () ->
            state <- logScopeExit scopeKind state
        )
    
    member _.LogTypeResolution(expr, resolvedType, location) =
        lock lockObj (fun () ->
            state <- logTypeResolution expr resolvedType location state
        )
    
    member _.LogPruningDecision(symbol, decision, reason) =
        lock lockObj (fun () ->
            state <- logPruningDecision symbol decision reason state
        )
    
    member _.GetState() =
        lock lockObj (fun () -> state)
    
    member _.GetEvents() =
        lock lockObj (fun () -> getEvents state)
    
    member _.GenerateDependencyGraph() =
        lock lockObj (fun () -> generateDependencyGraph state)
    
    member _.GenerateResolutionReport() =
        lock lockObj (fun () -> generateResolutionReport state)
    
    member _.GenerateStatistics() =
        lock lockObj (fun () -> generateStatistics state)
    
    member _.WriteToFile(filePath) =
        lock lockObj (fun () -> writeToFile filePath state)

/// Global diagnostics instance for the current compilation
let mutable private currentDiagnostics : DiagnosticsCollector option = None

/// Initialize diagnostics for a compilation run
let initializeDiagnostics() =
    currentDiagnostics <- Some (DiagnosticsCollector())

/// Get current diagnostics instance
let getDiagnostics() =
    match currentDiagnostics with
    | Some diag -> diag
    | None -> 
        let diag = DiagnosticsCollector()
        currentDiagnostics <- Some diag
        diag

/// Clear diagnostics
let clearDiagnostics() =
    currentDiagnostics <- None