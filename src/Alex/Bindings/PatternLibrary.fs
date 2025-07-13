module Alex.Bindings.PatternLibrary

open FSharp.Compiler.Syntax
open Core.Types.Dialects
open Core.Types.TypeSystem  // Use Core types since TypeMapping comes after this file



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

/// Core F# functions that need patterns
let coreFSharpPatterns = [
    // String formatting
    { Name = "sprintf"
      QualifiedName = "Microsoft.FSharp.Core.Printf.sprintf"
      OpPattern = Transform("sprintf_format", ["varargs"])
      TypeSig = ([MLIRTypes.memref MLIRTypes.i8], MLIRTypes.memref MLIRTypes.i8)
      Matcher = Matchers.applicationOf ["sprintf"] }
    
    // Array operations
    { Name = "Array.zeroCreate"
      QualifiedName = "Microsoft.FSharp.Collections.Array.zeroCreate"
      OpPattern = Transform("array_alloc_zero", ["size_param"])
      TypeSig = ([MLIRTypes.i32], MLIRTypes.memref MLIRTypes.i8)
      Matcher = Matchers.applicationOf ["Array.zeroCreate"; "zeroCreate"] }
    
    { Name = "Array.create"
      QualifiedName = "Microsoft.FSharp.Collections.Array.create"
      OpPattern = Transform("array_alloc_init", ["size_param"; "init_value"])
      TypeSig = ([MLIRTypes.i32; MLIRTypes.i32], MLIRTypes.memref MLIRTypes.i32)
      Matcher = Matchers.applicationOf ["Array.create"; "create"] }
    
    // Constructors
    { Name = "string_ctor"
      QualifiedName = "System.String..ctor"
      OpPattern = Transform("string_from_chars", ["char_array"])
      TypeSig = ([MLIRTypes.memref MLIRTypes.i8], MLIRTypes.memref MLIRTypes.i8)
      Matcher = function
        | SynExpr.New(_, SynType.LongIdent(SynLongIdent([ident], _, _)), _, _) 
            when ident.idText = "string" -> true
        | _ -> false }
    
    // Core functions
    { Name = "char"
      QualifiedName = "Microsoft.FSharp.Core.Operators.char"
      OpPattern = Transform("byte_to_char", ["cast"])
      TypeSig = ([MLIRTypes.i8], MLIRTypes.i8)
      Matcher = Matchers.applicationOf ["char"] }
]

// Additional F# core operators and functions
let fsharpCoreOperators = [
    // Pipe operators
    { Name = "op_PipeRight"
      QualifiedName = "Microsoft.FSharp.Core.Operators.op_PipeRight"
      OpPattern = Transform("identity", [])  // Pipe is just function application
      TypeSig = ([MLIRTypes.any; MLIRTypes.func [MLIRTypes.any] MLIRTypes.any], MLIRTypes.any)
      Matcher = Matchers.applicationOf ["|>"; "op_PipeRight"] }
    
    { Name = "op_PipeLeft"
      QualifiedName = "Microsoft.FSharp.Core.Operators.op_PipeLeft"
      OpPattern = Transform("reverse_apply", [])
      TypeSig = ([MLIRTypes.func [MLIRTypes.any] MLIRTypes.any; MLIRTypes.any], MLIRTypes.any)
      Matcher = Matchers.applicationOf ["<|"; "op_PipeLeft"] }
    
    // Comparison operators
    { Name = "op_Equality"
      QualifiedName = "Microsoft.FSharp.Core.Operators.op_Equality"
      OpPattern = DialectOp(Arith, "cmpi", Map["predicate", "eq"])
      TypeSig = ([MLIRTypes.i32; MLIRTypes.i32], MLIRTypes.i1)
      Matcher = Matchers.applicationOf ["="; "op_Equality"] }
    
    { Name = "op_Inequality"
      QualifiedName = "Microsoft.FSharp.Core.Operators.op_Inequality"
      OpPattern = DialectOp(Arith, "cmpi", Map["predicate", "ne"])
      TypeSig = ([MLIRTypes.i32; MLIRTypes.i32], MLIRTypes.i1)
      Matcher = Matchers.applicationOf ["<>"; "op_Inequality"] }
    
    { Name = "op_LessThan"
      QualifiedName = "Microsoft.FSharp.Core.Operators.op_LessThan"
      OpPattern = DialectOp(Arith, "cmpi", Map["predicate", "slt"])
      TypeSig = ([MLIRTypes.i32; MLIRTypes.i32], MLIRTypes.i1)
      Matcher = Matchers.applicationOf ["<"; "op_LessThan"] }
    
    { Name = "op_GreaterThan"
      QualifiedName = "Microsoft.FSharp.Core.Operators.op_GreaterThan"
      OpPattern = DialectOp(Arith, "cmpi", Map["predicate", "sgt"])
      TypeSig = ([MLIRTypes.i32; MLIRTypes.i32], MLIRTypes.i1)
      Matcher = Matchers.applicationOf [">"; "op_GreaterThan"] }
    
    // Arithmetic operators
    { Name = "op_Addition"
      QualifiedName = "Microsoft.FSharp.Core.Operators.op_Addition"
      OpPattern = DialectOp(Arith, "addi", Map.empty)
      TypeSig = ([MLIRTypes.i32; MLIRTypes.i32], MLIRTypes.i32)
      Matcher = Matchers.applicationOf ["+"; "op_Addition"] }
    
    { Name = "op_Subtraction"
      QualifiedName = "Microsoft.FSharp.Core.Operators.op_Subtraction"
      OpPattern = DialectOp(Arith, "subi", Map.empty)
      TypeSig = ([MLIRTypes.i32; MLIRTypes.i32], MLIRTypes.i32)
      Matcher = Matchers.applicationOf ["-"; "op_Subtraction"] }
    
    { Name = "op_Multiply"
      QualifiedName = "Microsoft.FSharp.Core.Operators.op_Multiply"
      OpPattern = DialectOp(Arith, "muli", Map.empty)
      TypeSig = ([MLIRTypes.i32; MLIRTypes.i32], MLIRTypes.i32)
      Matcher = Matchers.applicationOf ["*"; "op_Multiply"] }
    
    { Name = "op_Division"
      QualifiedName = "Microsoft.FSharp.Core.Operators.op_Division"
      OpPattern = DialectOp(Arith, "divsi", Map.empty)
      TypeSig = ([MLIRTypes.i32; MLIRTypes.i32], MLIRTypes.i32)
      Matcher = Matchers.applicationOf ["/"; "op_Division"] }
]

// Additional F# core functions
let fsharpCoreFunctions = [
    // Printf family
    { Name = "printf"
      QualifiedName = "Microsoft.FSharp.Core.ExtraTopLevelOperators.printf"
      OpPattern = ExternalCall("printf", Some "libc")
      TypeSig = ([MLIRTypes.memref MLIRTypes.i8], MLIRTypes.void_)
      Matcher = Matchers.applicationOf ["printf"] }
    
    { Name = "printfn"
      QualifiedName = "Microsoft.FSharp.Core.ExtraTopLevelOperators.printfn"
      OpPattern = Transform("printf_newline", ["format_string"])
      TypeSig = ([MLIRTypes.memref MLIRTypes.i8], MLIRTypes.void_)
      Matcher = Matchers.applicationOf ["printfn"] }
    
    // List operations
    { Name = "List.map"
      QualifiedName = "Microsoft.FSharp.Collections.ListModule.map"
      OpPattern = Transform("list_map", ["mapper_function"])
      TypeSig = ([MLIRTypes.func [MLIRTypes.any] MLIRTypes.any; MLIRTypes.memref MLIRTypes.any], 
                 MLIRTypes.memref MLIRTypes.any)
      Matcher = Matchers.applicationOf ["List.map"; "map"] }
    
    { Name = "List.filter"
      QualifiedName = "Microsoft.FSharp.Collections.ListModule.filter"
      OpPattern = Transform("list_filter", ["predicate_function"])
      TypeSig = ([MLIRTypes.func [MLIRTypes.any] MLIRTypes.i1; MLIRTypes.memref MLIRTypes.any], 
                 MLIRTypes.memref MLIRTypes.any)
      Matcher = Matchers.applicationOf ["List.filter"; "filter"] }
    
    // Option operations
    { Name = "Some"
      QualifiedName = "Microsoft.FSharp.Core.Option.Some"
      OpPattern = Transform("option_some", ["value"])
      TypeSig = ([MLIRTypes.any], MLIRTypes.opaque "option")
      Matcher = Matchers.applicationOf ["Some"] }
    
    { Name = "None"
      QualifiedName = "Microsoft.FSharp.Core.Option.None"
      OpPattern = Transform("option_none", [])
      TypeSig = ([], MLIRTypes.opaque "option")
      Matcher = Matchers.applicationOf ["None"] }
]

// Combine all patterns
let allPatterns = alloyPatterns @ coreFSharpPatterns @ fsharpCoreOperators @ fsharpCoreFunctions

/// Find pattern by qualified name
let findByName qualifiedName =
    allPatterns |> List.tryFind (fun p -> 
        p.QualifiedName = qualifiedName || 
        qualifiedName.EndsWith(p.QualifiedName))

/// Find pattern by expression
let findByExpression expr =
    allPatterns |> List.tryFind (fun p -> p.Matcher expr)