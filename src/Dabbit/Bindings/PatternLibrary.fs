module Dabbit.Bindings.PatternLibrary

open FSharp.Compiler.Syntax
open Core.MLIRGeneration.TypeSystem
open Core.MLIRGeneration.Dialect

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
        | SynExpr.App(_, _, SynExpr.LongIdent(_, lid, _, _), _, _) ->
            match lid.Lid with
            | [ident] when ident.idText = name -> Some ()
            | _ -> None
        | _ -> None
    
    let (|StringLiteral|_|) = function
        | SynExpr.Const(SynConst.String(s, _, _), _) -> Some s
        | _ -> None
    
    let isResultReturning name =
        name = "readInto" || name = "readFile" || 
        name.StartsWith("try") || name.Contains("OrNone")
    
    let applicationOf names expr =
        names |> List.exists (fun name ->
            match expr with
            | AppNamed name -> true
            | _ -> false)

/// Core Alloy library patterns
let alloyPatterns : SymbolPattern list = [
    // Stack allocation
    { Name = "stack-buffer"
      QualifiedName = "Alloy.Memory.stackBuffer"
      OpPattern = DialectOp(MemRef, "memref.alloca", Map["element_type", "i8"])
      TypeSig = ([MLIRTypes.i32], MLIRTypes.memref MLIRTypes.i8)
      Matcher = Matchers.applicationOf ["stackBuffer"] }
    
    // Native pointer allocation
    { Name = "nativeptr-stackalloc"
      QualifiedName = "NativePtr.stackalloc"
      OpPattern = DialectOp(MemRef, "memref.alloca", Map["element_type", "i8"])
      TypeSig = ([MLIRTypes.i32], MLIRTypes.memref MLIRTypes.i8)
      Matcher = Matchers.applicationOf ["NativePtr.stackalloc"] }
    
    // String formatting
    { Name = "string-format"
      QualifiedName = "Alloy.IO.String.format"
      OpPattern = Composite [
          ExternalCall("sprintf", Some "libc")
          Transform("utf8_conversion", [])
      ]
      TypeSig = ([MLIRTypes.memref MLIRTypes.i8; MLIRTypes.memref MLIRTypes.i8], 
                 MLIRTypes.memref MLIRTypes.i8)
      Matcher = fun expr ->
          match expr with
          | SynExpr.App(_, _, Matchers.AppNamed "format", 
                        SynExpr.Tuple(_, [Matchers.StringLiteral _; _], _, _), _) -> true
          | _ -> false }
    
    // Console operations
    { Name = "console-writeline"
      QualifiedName = "Alloy.IO.Console.writeLine"
      OpPattern = ExternalCall("printf", Some "libc")
      TypeSig = ([MLIRTypes.memref MLIRTypes.i8], MLIRTypes.void_)
      Matcher = Matchers.applicationOf ["writeLine"] }
    
    { Name = "console-readinto"
      QualifiedName = "Alloy.IO.Console.readInto"
      OpPattern = Composite [
          ExternalCall("fgets", Some "libc")
          ExternalCall("strlen", Some "libc")
          Transform("result_wrapper", ["success_check"])
      ]
      TypeSig = ([MLIRTypes.memref MLIRTypes.i8], MLIRTypes.i32)
      Matcher = Matchers.applicationOf ["readInto"] }
]

/// Find pattern by qualified name
let findByName qualifiedName =
    alloyPatterns |> List.tryFind (fun p -> 
        p.QualifiedName = qualifiedName || 
        qualifiedName.EndsWith(p.QualifiedName))

/// Find pattern by expression
let findByExpression expr =
    alloyPatterns |> List.tryFind (fun p -> p.Matcher expr)