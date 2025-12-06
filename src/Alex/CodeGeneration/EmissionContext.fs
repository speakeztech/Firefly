/// EmissionContext - SSA tracking and MLIR output building
///
/// This module provides the foundational types for MLIR code generation:
/// - SSAContext: Tracks SSA values, locals, and string literals
/// - MLIRBuilder: StringBuilder-based MLIR text output
///
/// Extracted from Core.MLIR.Emitter to support the Alex architecture.
module Alex.CodeGeneration.EmissionContext

open System
open System.Text

/// SSA value counter and local variable tracking
///
/// Maintains mappings from F# names to SSA values (%v0, %v1, etc.)
/// and tracks types for FIDELITY verification.
type SSAContext = {
    mutable Counter: int
    mutable Locals: Map<string, string>       // F# name -> SSA name
    mutable LocalTypes: Map<string, string>   // F# name -> MLIR type
    mutable SSATypes: Map<string, string>     // SSA name -> MLIR type
    mutable StringLiterals: (string * string) list  // content -> global name
}

module SSAContext =
    /// Create a fresh SSA context
    let create () : SSAContext = {
        Counter = 0
        Locals = Map.empty
        LocalTypes = Map.empty
        SSATypes = Map.empty
        StringLiterals = []
    }

    /// Generate the next SSA value name
    let nextValue (ctx: SSAContext) : string =
        let name = sprintf "%%v%d" ctx.Counter
        ctx.Counter <- ctx.Counter + 1
        name

    /// Generate next SSA value and track its type
    let nextValueWithType (ctx: SSAContext) (mlirType: string) : string =
        let name = sprintf "%%v%d" ctx.Counter
        ctx.Counter <- ctx.Counter + 1
        ctx.SSATypes <- Map.add name mlirType ctx.SSATypes
        name

    /// Create a new local binding
    let bindLocal (ctx: SSAContext) (fsharpName: string) : string =
        let ssaName = nextValue ctx
        ctx.Locals <- Map.add fsharpName ssaName ctx.Locals
        ssaName

    /// Lookup a local variable's SSA name
    let lookupLocal (ctx: SSAContext) (name: string) : string option =
        Map.tryFind name ctx.Locals

    /// Lookup a local variable's MLIR type
    let lookupLocalType (ctx: SSAContext) (name: string) : string option =
        Map.tryFind name ctx.LocalTypes

    /// Lookup an SSA value's MLIR type
    let lookupSSAType (ctx: SSAContext) (ssaName: string) : string option =
        Map.tryFind ssaName ctx.SSATypes

    /// Register a local variable with its SSA name
    let registerLocal (ctx: SSAContext) (fsharpName: string) (ssaName: string) : unit =
        ctx.Locals <- Map.add fsharpName ssaName ctx.Locals

    /// Register a local variable with both SSA name and type
    let registerLocalWithType (ctx: SSAContext) (fsharpName: string) (ssaName: string) (mlirType: string) : unit =
        ctx.Locals <- Map.add fsharpName ssaName ctx.Locals
        ctx.LocalTypes <- Map.add fsharpName mlirType ctx.LocalTypes
        ctx.SSATypes <- Map.add ssaName mlirType ctx.SSATypes

    /// Register just the SSA value type
    let registerSSAType (ctx: SSAContext) (ssaName: string) (mlirType: string) : unit =
        ctx.SSATypes <- Map.add ssaName mlirType ctx.SSATypes

    /// Add or lookup a string literal, returning its global name
    let addStringLiteral (ctx: SSAContext) (content: string) : string =
        match ctx.StringLiterals |> List.tryFind (fun (c, _) -> c = content) with
        | Some (_, name) -> name
        | None ->
            let name = sprintf "@str%d" (List.length ctx.StringLiterals)
            ctx.StringLiterals <- (content, name) :: ctx.StringLiterals
            name

    /// Get all registered string literals
    let getStringLiterals (ctx: SSAContext) : (string * string) list =
        ctx.StringLiterals

/// MLIR text output builder
///
/// Provides indentation-aware output building for MLIR text format.
type MLIRBuilder = {
    mutable Indent: int
    Output: StringBuilder
}

module MLIRBuilder =
    /// Create a fresh builder
    let create () : MLIRBuilder = {
        Indent = 0
        Output = StringBuilder()
    }

    /// Get the current indentation string
    let indent (b: MLIRBuilder) : string =
        String.replicate (b.Indent * 2) " "

    /// Append a line with current indentation
    let line (b: MLIRBuilder) (text: string) : unit =
        b.Output.AppendLine(indent b + text) |> ignore

    /// Append a line without indentation
    let lineNoIndent (b: MLIRBuilder) (text: string) : unit =
        b.Output.AppendLine(text) |> ignore

    /// Append raw text (no newline)
    let raw (b: MLIRBuilder) (text: string) : unit =
        b.Output.Append(text) |> ignore

    /// Increase indentation level
    let push (b: MLIRBuilder) : unit =
        b.Indent <- b.Indent + 1

    /// Decrease indentation level
    let pop (b: MLIRBuilder) : unit =
        b.Indent <- b.Indent - 1

    /// Get the accumulated output
    let toString (b: MLIRBuilder) : string =
        b.Output.ToString()

    /// Clear and reset the builder
    let clear (b: MLIRBuilder) : unit =
        b.Output.Clear() |> ignore
        b.Indent <- 0

    /// Execute an action with increased indentation
    let withIndent (b: MLIRBuilder) (action: unit -> unit) : unit =
        push b
        action ()
        pop b

    /// Build a block structure: { ... }
    let block (b: MLIRBuilder) (header: string) (body: unit -> unit) : unit =
        line b (header + " {")
        withIndent b body
        line b "}"

/// Combined context for emission
///
/// Bundles SSAContext and MLIRBuilder for convenient passing.
type EmissionEnv = {
    SSA: SSAContext
    Builder: MLIRBuilder
}

module EmissionEnv =
    /// Create a fresh emission environment
    let create () : EmissionEnv = {
        SSA = SSAContext.create ()
        Builder = MLIRBuilder.create ()
    }

    /// Generate next SSA value
    let nextSSA (env: EmissionEnv) : string =
        SSAContext.nextValue env.SSA

    /// Generate next SSA value with type
    let nextSSAWithType (mlirType: string) (env: EmissionEnv) : string =
        SSAContext.nextValueWithType env.SSA mlirType

    /// Emit a line
    let emit (text: string) (env: EmissionEnv) : unit =
        MLIRBuilder.line env.Builder text

    /// Get the output
    let output (env: EmissionEnv) : string =
        MLIRBuilder.toString env.Builder
