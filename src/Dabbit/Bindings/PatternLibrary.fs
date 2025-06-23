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
        | SynExpr.App(_, _, SynExpr.LongIdent(_, SynLongIdent(ids, _, _), _, _), _, _) ->
            match ids with
            | [ident] when ident.idText = name -> Some ()
            | ids -> 
                let fullName = ids |> List.map (fun i -> i.idText) |> String.concat "."
                if fullName.EndsWith(name) then Some () else None
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
          | Matchers.AppNamed "format" -> true
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
    
    // Span operations
    { Name = "span-to-string"
      QualifiedName = "Alloy.Memory.spanToString"
      OpPattern = Transform("span_to_string", ["utf8_conversion"])
      TypeSig = ([MLIRTypes.memref MLIRTypes.i8], MLIRTypes.memref MLIRTypes.i8)
      Matcher = Matchers.applicationOf ["spanToString"] }
    
    // Buffer operations
    { Name = "buffer-as-span"
      QualifiedName = "Alloy.Memory.BufferOps.AsSpan"
      OpPattern = Transform("buffer_slice", ["offset"; "length"])
      TypeSig = ([MLIRTypes.memref MLIRTypes.i8; MLIRTypes.i32; MLIRTypes.i32], 
                 MLIRTypes.memref MLIRTypes.i8)
      Matcher = fun expr ->
          match expr with
          | SynExpr.DotGet(_, _, SynLongIdent([ident], _, _), _) when ident.idText = "AsSpan" -> true
          | _ -> false }
    
    // Console prompt
    { Name = "console-prompt"
      QualifiedName = "Alloy.IO.Console.prompt"
      OpPattern = ExternalCall("printf", Some "libc")
      TypeSig = ([MLIRTypes.memref MLIRTypes.i8], MLIRTypes.void_)
      Matcher = Matchers.applicationOf ["prompt"] }
    
    // Result operations
    { Name = "result-ok"
      QualifiedName = "Alloy.Core.Result.Ok"
      OpPattern = Transform("result_ok", ["tag_value"])
      TypeSig = ([MLIRTypes.i32], 
                 MLIRTypes.struct_([MLIRTypes.i32; MLIRTypes.i32]))
      Matcher = fun expr ->
          match expr with
          | Matchers.AppNamed "Ok" -> true
          | _ -> false }
    
    { Name = "result-error"
      QualifiedName = "Alloy.Core.Result.Error"
      OpPattern = Transform("result_error", ["tag_value"])
      TypeSig = ([MLIRTypes.i32], 
                 MLIRTypes.struct_([MLIRTypes.i32; MLIRTypes.i32]))
      Matcher = fun expr ->
          match expr with
          | Matchers.AppNamed "Error" -> true
          | _ -> false }
    
    // Core operations
    { Name = "core-iter"
      QualifiedName = "Alloy.Core.iter"
      OpPattern = Transform("iterate_array", ["inline_function"])
      TypeSig = ([(MLIRTypes.func [MLIRTypes.i8] MLIRTypes.void_); MLIRTypes.memref MLIRTypes.i8], 
                 MLIRTypes.void_)
      Matcher = Matchers.applicationOf ["iter"] }
    
    { Name = "core-map"
      QualifiedName = "Alloy.Core.map"
      OpPattern = Transform("map_array", ["inline_function"])
      TypeSig = ([(MLIRTypes.func [MLIRTypes.i8] MLIRTypes.i8); MLIRTypes.memref MLIRTypes.i8], 
                 MLIRTypes.memref MLIRTypes.i8)
      Matcher = Matchers.applicationOf ["map"] }
    
    { Name = "core-len"
      QualifiedName = "Alloy.Core.len"
      OpPattern = Transform("array_length", [])
      TypeSig = ([MLIRTypes.memref MLIRTypes.i8], MLIRTypes.i32)
      Matcher = Matchers.applicationOf ["len"] }
    
    { Name = "zero-value"
      QualifiedName = "Alloy.Core.zero"
      OpPattern = Transform("zero_value", ["type_dependent"])
      TypeSig = ([], MLIRTypes.i32)
      Matcher = Matchers.applicationOf ["zero"] }
    
    { Name = "one-value"
      QualifiedName = "Alloy.Core.one"
      OpPattern = Transform("one_value", ["type_dependent"])
      TypeSig = ([], MLIRTypes.i32)
      Matcher = Matchers.applicationOf ["one"] }
    
    { Name = "not-bool"
      QualifiedName = "Alloy.Core.not"
      OpPattern = DialectOp(Arith, "arith.xori", Map["rhs", "1"])
      TypeSig = ([MLIRTypes.i1], MLIRTypes.i1)
      Matcher = Matchers.applicationOf ["not"] }
    
    // String operations
    { Name = "string-concat"
      QualifiedName = "Alloy.IO.String.concat"
      OpPattern = ExternalCall("strcat", Some "libc")
      TypeSig = ([MLIRTypes.memref MLIRTypes.i8; MLIRTypes.memref MLIRTypes.i8], 
                 MLIRTypes.memref MLIRTypes.i8)
      Matcher = Matchers.applicationOf ["concat"] }
    
    { Name = "string-length"
      QualifiedName = "Alloy.IO.String.length"
      OpPattern = ExternalCall("strlen", Some "libc")
      TypeSig = ([MLIRTypes.memref MLIRTypes.i8], MLIRTypes.i32)
      Matcher = Matchers.applicationOf ["length"] }
    
    // Arithmetic operations
    { Name = "add-op"
      QualifiedName = "Alloy.Numerics.add"
      OpPattern = Transform("add_resolved", ["type_dependent"])
      TypeSig = ([MLIRTypes.i32; MLIRTypes.i32], MLIRTypes.i32)
      Matcher = Matchers.applicationOf ["add"; "+"] }
    
    { Name = "subtract-op"
      QualifiedName = "Alloy.Numerics.subtract"
      OpPattern = Transform("subtract_resolved", ["type_dependent"])
      TypeSig = ([MLIRTypes.i32; MLIRTypes.i32], MLIRTypes.i32)
      Matcher = Matchers.applicationOf ["subtract"; "-"] }
    
    { Name = "multiply-op"
      QualifiedName = "Alloy.Numerics.multiply"
      OpPattern = Transform("multiply_resolved", ["type_dependent"])
      TypeSig = ([MLIRTypes.i32; MLIRTypes.i32], MLIRTypes.i32)
      Matcher = Matchers.applicationOf ["multiply"; "*"] }
    
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
      TypeSig = ([MLIRTypes.i32; (MLIRTypes.func [MLIRTypes.i32] MLIRTypes.i32)], 
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
      TypeSig = ([(MLIRTypes.func [MLIRTypes.i32] MLIRTypes.i32);
                  (MLIRTypes.func [MLIRTypes.i32] MLIRTypes.i32)], 
                 (MLIRTypes.func [MLIRTypes.i32] MLIRTypes.i32))
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