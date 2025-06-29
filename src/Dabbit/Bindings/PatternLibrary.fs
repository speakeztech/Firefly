module Dabbit.Bindings.PatternLibrary

open FSharp.Compiler.Syntax
open Core.Types.TypeSystem  // Use Core types since TypeMapping comes after this file

/// MLIR operation pattern for resolved symbols
type MLIROperationPattern =
    | DialectOp of dialect: MLIRDialect * op: string * attrs: Map<string, string>
    | ExternalCall of func: string * lib: string option
    | Composite of MLIROperationPattern list
    | Transform of name: string * parameters: string list

/// Pattern matcher for FCS syntax expressions
type ExprMatcher = SynExpr -> bool

/// Symbol pattern for pattern-based transformations
type SymbolPattern = {
    Name: string
    QualifiedName: string
    OpPattern: MLIROperationPattern
    TypeSig: MLIRType list * MLIRType
    Matcher: ExprMatcher
}

/// Common expression pattern matchers
module Matchers =
    let (|AppNamed|_|) name = function
        | SynExpr.App(_, _, SynExpr.Ident(ident), _, _) when ident.idText = name -> Some ()
        | SynExpr.App(_, _, SynExpr.LongIdent(_, SynLongIdent(ids, _, _), _, _), _, _) ->
            match ids with
            | [ident] when ident.idText = name -> Some ()
            | ids -> 
                let fullName = ids |> List.map (fun i -> i.idText) |> String.concat "."
                if fullName.EndsWith("." + name) then Some () else None
        | _ -> None
    
    let applicationOf names expr =
        names |> List.exists (fun name -> 
            match expr with
            | AppNamed name -> true
            | _ -> false)

/// Alloy standard library patterns
let alloyPatterns = [
    // Stack allocation
    { Name = "stackBuffer"
      QualifiedName = "Alloy.Memory.stackBuffer"
      OpPattern = DialectOp(MLIRDialect.MemRef, "alloca", Map["element_type", "i8"])  // Use MLIRDialect.MemRef
      TypeSig = ([MLIRTypes.i32], MLIRTypes.memref MLIRTypes.i8)
      Matcher = Matchers.applicationOf ["stackBuffer"] }
    
    // Buffer operations
    { Name = "salloc"
      QualifiedName = "Alloy.Memory.salloc"
      OpPattern = DialectOp(MLIRDialect.MemRef, "alloca", Map["static", "true"])  // Use MLIRDialect.MemRef
      TypeSig = ([MLIRTypes.i32], MLIRTypes.memref MLIRTypes.i8)
      Matcher = Matchers.applicationOf ["salloc"] }
    
    // Console I/O
    { Name = "write"
      QualifiedName = "Alloy.IO.Console.write"
      OpPattern = ExternalCall("printf", Some "libc")
      TypeSig = ([MLIRTypes.memref MLIRTypes.i8], MLIRTypes.void_)
      Matcher = Matchers.applicationOf ["write"; "Console.write"] }
    
    { Name = "writeLine"
      QualifiedName = "Alloy.IO.Console.writeLine"
      OpPattern = ExternalCall("puts", Some "libc")
      TypeSig = ([MLIRTypes.memref MLIRTypes.i8], MLIRTypes.void_)
      Matcher = Matchers.applicationOf ["writeLine"; "Console.writeLine"] }
    
    // String operations
    { Name = "format"
      QualifiedName = "Alloy.IO.String.format"
      OpPattern = ExternalCall("sprintf", Some "libc")
      TypeSig = ([MLIRTypes.memref MLIRTypes.i8; MLIRTypes.memref MLIRTypes.i8], 
                 MLIRTypes.memref MLIRTypes.i8)
      Matcher = Matchers.applicationOf ["format"; "String.format"] }
    
    // Basic arithmetic
    { Name = "add-op"
      QualifiedName = "Alloy.Numerics.add"
      OpPattern = DialectOp(Arith, "addi", Map.empty)
      TypeSig = ([MLIRTypes.i32; MLIRTypes.i32], MLIRTypes.i32)
      Matcher = Matchers.applicationOf ["add"; "+"] }
    
    { Name = "sub-op"
      QualifiedName = "Alloy.Numerics.sub"
      OpPattern = DialectOp(Arith, "subi", Map.empty)
      TypeSig = ([MLIRTypes.i32; MLIRTypes.i32], MLIRTypes.i32)
      Matcher = Matchers.applicationOf ["sub"; "-"] }
    
    { Name = "mul-op"
      QualifiedName = "Alloy.Numerics.mul"
      OpPattern = DialectOp(Arith, "muli", Map.empty)
      TypeSig = ([MLIRTypes.i32; MLIRTypes.i32], MLIRTypes.i32)
      Matcher = Matchers.applicationOf ["mul"; "*"] }
    
    { Name = "div-op"
      QualifiedName = "Alloy.Numerics.div"
      OpPattern = DialectOp(Arith, "divsi", Map.empty)
      TypeSig = ([MLIRTypes.i32; MLIRTypes.i32], MLIRTypes.i32)
      Matcher = Matchers.applicationOf ["div"; "/"] }
    
    // Comparison operations
    { Name = "equals-op"
      QualifiedName = "Alloy.Numerics.equals"
      OpPattern = Transform("equals_resolved", ["type_dependent"])
      TypeSig = ([MLIRTypes.i32; MLIRTypes.i32], MLIRTypes.i1)
      Matcher = Matchers.applicationOf ["equals"; "="] }
    
    { Name = "less-than-op"
      QualifiedName = "Alloy.Numerics.lessThan"
      OpPattern = Transform("less_than_resolved", ["type_dependent"])
      TypeSig = ([MLIRTypes.i32; MLIRTypes.i32], MLIRTypes.i1)
      Matcher = Matchers.applicationOf ["lessThan"; "<"] }
    
    // Pipe operations
    { Name = "pipe-right"
      QualifiedName = "Alloy.Operators.op_PipeRight"
      OpPattern = Transform("pipe_right", ["function_application"])
      TypeSig = ([MLIRTypes.i32; MLIRTypes.func [MLIRTypes.i32] MLIRTypes.i32], 
                 MLIRTypes.i32)
      Matcher = fun expr ->
          match expr with
          | SynExpr.App(_, _, SynExpr.App(_, _, SynExpr.Ident(op), _, _), _, _) 
              when op.idText = "op_PipeRight" || op.idText = "|>" -> true
          | _ -> false }
    
    // Function composition
    { Name = "compose-right"
      QualifiedName = "Alloy.Operators.op_ComposeRight"
      OpPattern = Transform("compose_right", ["function_composition"])
      TypeSig = ([MLIRTypes.func [MLIRTypes.i32] MLIRTypes.i32;
                  MLIRTypes.func [MLIRTypes.i32] MLIRTypes.i32], 
                 MLIRTypes.func [MLIRTypes.i32] MLIRTypes.i32)
      Matcher = Matchers.applicationOf [">>"] }
]

/// Find pattern by qualified name
let findByName qualifiedName =
    alloyPatterns |> List.tryFind (fun p -> 
        p.QualifiedName = qualifiedName || 
        qualifiedName.EndsWith(p.QualifiedName))

/// Find pattern by expression
let findByExpression expr =
    alloyPatterns |> List.tryFind (fun p -> p.Matcher expr)