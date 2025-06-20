module Dabbit.TreeShaking.ReachabilityAnalyzer

open System
open System.Collections.Generic
open Core.XParsec.Foundation
open Dabbit.Parsing.OakAst

/// Represents a dependency relationship in the program
type Dependency =
    | FunctionCall of caller: string * callee: string
    | TypeUsage of user: string * typeName: string
    | UnionCaseUsage of user: string * typeName: string * caseName: string
    | FieldAccess of user: string * typeName: string * fieldName: string
    | ModuleReference of user: string * moduleName: string

/// Dependency graph for reachability analysis
type DependencyGraph = {
    /// Map from declaration name to its dependencies
    Dependencies: Map<string, Set<Dependency>>
    /// Map from declaration name to its AST node
    Declarations: Map<string, OakDeclaration>
    /// Entry points (main function, exported APIs)
    EntryPoints: Set<string>
    /// Module-qualified names
    QualifiedNames: Map<string, string>
}

/// Reachability analysis result
type ReachabilityResult = {
    /// Set of reachable declaration names
    ReachableDeclarations: Set<string>
    /// Set of reachable type cases (for unions)
    ReachableUnionCases: Map<string, Set<string>>
    /// Set of reachable record fields
    ReachableFields: Map<string, Set<string>>
    /// Diagnostics about eliminated code
    EliminationStats: EliminationStatistics
}

/// Statistics about eliminated code
and EliminationStatistics = {
    TotalDeclarations: int
    ReachableDeclarations: int
    EliminatedDeclarations: int
    ModuleBreakdown: Map<string, ModuleStats>
    LargestEliminated: (string * int) list
}

and ModuleStats = {
    ModuleName: string
    TotalFunctions: int
    RetainedFunctions: int
    EliminatedFunctions: int
}