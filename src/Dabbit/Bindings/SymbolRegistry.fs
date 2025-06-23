module Dabbit.Bindings.SymbolRegistry

open System
open FSharp.Compiler.Syntax
open Core.XParsec.Foundation
open Core.MLIRGeneration.TypeSystem
open Core.MLIRGeneration.Dialect
open Dabbit.Bindings.PatternLibrary

/// Symbol with type safety and MLIR generation info
type ResolvedSymbol = {
    QualifiedName: string
    ShortName: string
    ParameterTypes: MLIRType list
    ReturnType: MLIRType
    Operation: MLIROperationPattern
    Namespace: string
    SourceLibrary: string
    RequiresExternal: bool
}

/// Symbol resolution state
type SymbolResolutionState = {
    SymbolsByQualified: Map<string, ResolvedSymbol>
    SymbolsByShort: Map<string, ResolvedSymbol>
    NamespaceMap: Map<string, string list>
    ActiveNamespaces: string list
    TypeRegistry: Map<string, MLIRType>
    ActiveSymbols: string list
    PatternRegistry: SymbolPattern list
}

/// The complete symbol registry
type SymbolRegistry = {
    State: SymbolResolutionState
    ResolutionHistory: (string * string) list
}

/// Symbol resolution functions
module SymbolResolution =
    let createEmptyState () : SymbolResolutionState = {
        SymbolsByQualified = Map.empty
        SymbolsByShort = Map.empty
        NamespaceMap = Map.empty
        ActiveNamespaces = []
        TypeRegistry = Map.empty
        ActiveSymbols = []
        PatternRegistry = alloyPatterns
    }
    
    let registerSymbol (symbol: ResolvedSymbol) (state: SymbolResolutionState) : SymbolResolutionState =
        let namespaceSymbols = 
            match Map.tryFind symbol.Namespace state.NamespaceMap with
            | Some existing -> symbol.ShortName :: existing
            | None -> [symbol.ShortName]
            
        { state with
            SymbolsByQualified = Map.add symbol.QualifiedName symbol state.SymbolsByQualified
            SymbolsByShort = Map.add symbol.ShortName symbol state.SymbolsByShort
            NamespaceMap = Map.add symbol.Namespace namespaceSymbols state.NamespaceMap }
    
    let resolveQualified (qualifiedName: string) (state: SymbolResolutionState) : CompilerResult<ResolvedSymbol> =
        match Map.tryFind qualifiedName state.SymbolsByQualified with
        | Some symbol -> Success symbol
        | None -> 
            match findByName qualifiedName with
            | Some pattern ->
                let symbol = {
                    QualifiedName = qualifiedName
                    ShortName = 
                        if qualifiedName.Contains(".") then
                            qualifiedName.Split([|'.'|], StringSplitOptions.RemoveEmptyEntries) |> Array.last
                        else qualifiedName
                    ParameterTypes = fst pattern.TypeSig
                    ReturnType = snd pattern.TypeSig
                    Operation = pattern.OpPattern
                    Namespace = 
                        if qualifiedName.Contains(".") then
                            qualifiedName.Substring(0, qualifiedName.LastIndexOf('.'))
                        else ""
                    SourceLibrary = "Alloy"
                    RequiresExternal = 
                        match pattern.OpPattern with
                        | ExternalCall(_, Some _) -> true
                        | Composite patterns ->
                            patterns |> List.exists (function
                                | ExternalCall(_, Some _) -> true
                                | _ -> false)
                        | _ -> false
                }
                Success symbol
            | None ->
                CompilerFailure [ConversionError(
                    "qualified symbol resolution",
                    qualifiedName,
                    "resolved symbol",
                    sprintf "Symbol '%s' not found in qualified registry or pattern library" qualifiedName
                )]
    
    let resolveUnqualified (shortName: string) (state: SymbolResolutionState) : CompilerResult<ResolvedSymbol> =
        match Map.tryFind shortName state.SymbolsByShort with
        | Some symbol -> Success symbol
        | None ->
            let tryNamespaceResolution () =
                state.ActiveNamespaces
                |> List.tryPick (fun ns ->
                    let qualifiedAttempt = sprintf "%s.%s" ns shortName
                    Map.tryFind qualifiedAttempt state.SymbolsByQualified)
            
            match tryNamespaceResolution () with
            | Some symbol -> Success symbol
            | None ->
                CompilerFailure [ConversionError(
                    "unqualified symbol resolution",
                    shortName,
                    "resolved symbol",
                    sprintf "Symbol '%s' not found in any active namespace" shortName
                )]
    
    let resolveSymbol (name: string) (state: SymbolResolutionState) : CompilerResult<ResolvedSymbol> =
        if name.Contains(".") 
        then resolveQualified name state
        else resolveUnqualified name state

/// Alloy library symbol definitions
module AlloySymbols =
    
    // Core module symbols
    let createCoreSymbols() : ResolvedSymbol list = [
        // Collection operations
        {
            QualifiedName = "Alloy.Core.iter"
            ShortName = "iter"
            ParameterTypes = [MLIRTypes.func([MLIRTypes.i8], MLIRTypes.void_); MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.void_
            Operation = Transform("iterate_array", ["inline_function"])
            Namespace = "Alloy.Core"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Core.map"
            ShortName = "map"
            ParameterTypes = [MLIRTypes.func([MLIRTypes.i8], MLIRTypes.i8); MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.memref MLIRTypes.i8
            Operation = Transform("map_array", ["inline_function"])
            Namespace = "Alloy.Core"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Core.len"
            ShortName = "len"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.i32
            Operation = Transform("array_length", [])
            Namespace = "Alloy.Core"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        // Zero/One/Default operations
        {
            QualifiedName = "Alloy.Core.zero"
            ShortName = "zero"
            ParameterTypes = []
            ReturnType = MLIRTypes.i32  // Will be resolved based on usage
            Operation = Transform("zero_value", ["type_dependent"])
            Namespace = "Alloy.Core"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Core.one"
            ShortName = "one"
            ParameterTypes = []
            ReturnType = MLIRTypes.i32  // Will be resolved based on usage
            Operation = Transform("one_value", ["type_dependent"])
            Namespace = "Alloy.Core"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        // Boolean operations
        {
            QualifiedName = "Alloy.Core.not"
            ShortName = "not"
            ParameterTypes = [MLIRTypes.i1]
            ReturnType = MLIRTypes.i1
            Operation = DialectOp(Arith, "arith.xori", Map["rhs", "1"])
            Namespace = "Alloy.Core"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
    ]
    
    // Numeric operations symbols
    let createNumericSymbols() : ResolvedSymbol list = [
        // Basic arithmetic - will be resolved to concrete types at usage
        {
            QualifiedName = "Alloy.Numerics.add"
            ShortName = "add"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            Operation = Transform("add_resolved", ["type_dependent"])
            Namespace = "Alloy.Numerics"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Numerics.subtract"
            ShortName = "subtract"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            Operation = Transform("subtract_resolved", ["type_dependent"])
            Namespace = "Alloy.Numerics"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Numerics.multiply"
            ShortName = "multiply"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            Operation = Transform("multiply_resolved", ["type_dependent"])
            Namespace = "Alloy.Numerics"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Numerics.divide"
            ShortName = "divide"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            Operation = Transform("divide_resolved", ["type_dependent"])
            Namespace = "Alloy.Numerics"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Numerics.modulo"
            ShortName = "modulo"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            Operation = Transform("modulo_resolved", ["type_dependent"])
            Namespace = "Alloy.Numerics"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Numerics.power"
            ShortName = "power"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            Operation = Transform("power_resolved", ["type_dependent"])
            Namespace = "Alloy.Numerics"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        // Mathematical functions
        {
            QualifiedName = "Alloy.Numerics.abs"
            ShortName = "abs"
            ParameterTypes = [MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            Operation = Transform("abs_resolved", ["type_dependent"])
            Namespace = "Alloy.Numerics"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Numerics.sqrt"
            ShortName = "sqrt"
            ParameterTypes = [MLIRTypes.f32]
            ReturnType = MLIRTypes.f32
            Operation = Transform("sqrt_resolved", ["type_dependent"])
            Namespace = "Alloy.Numerics"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        // Comparison operations
        {
            QualifiedName = "Alloy.Numerics.equals"
            ShortName = "equals"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i1
            Operation = Transform("equals_resolved", ["type_dependent"])
            Namespace = "Alloy.Numerics"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Numerics.lessThan"
            ShortName = "lessThan"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i1
            Operation = Transform("less_than_resolved", ["type_dependent"])
            Namespace = "Alloy.Numerics"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Numerics.greaterThan"
            ShortName = "greaterThan"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i1
            Operation = Transform("greater_than_resolved", ["type_dependent"])
            Namespace = "Alloy.Numerics"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Numerics.lessThanOrEqual"
            ShortName = "lessThanOrEqual"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i1
            Operation = Transform("less_than_or_equal_resolved", ["type_dependent"])
            Namespace = "Alloy.Numerics"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Numerics.greaterThanOrEqual"
            ShortName = "greaterThanOrEqual"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i1
            Operation = Transform("greater_than_or_equal_resolved", ["type_dependent"])
            Namespace = "Alloy.Numerics"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
    ]
    
    // Operator symbols
    let createOperatorSymbols() : ResolvedSymbol list = [
        // Pipe operators
        {
            QualifiedName = "Alloy.Operators.op_PipeRight"
            ShortName = "|>"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.func([MLIRTypes.i32], MLIRTypes.i32)]
            ReturnType = MLIRTypes.i32
            Operation = Transform("pipe_right", ["function_application"])
            Namespace = "Alloy.Operators"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Operators.op_PipeLeft"
            ShortName = "<|"
            ParameterTypes = [MLIRTypes.func([MLIRTypes.i32], MLIRTypes.i32); MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            Operation = Transform("pipe_left", ["function_application"])
            Namespace = "Alloy.Operators"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Operators.op_ComposeRight"
            ShortName = ">>"
            ParameterTypes = [
                MLIRTypes.func([MLIRTypes.i32], MLIRTypes.i32)
                MLIRTypes.func([MLIRTypes.i32], MLIRTypes.i32)
            ]
            ReturnType = MLIRTypes.func([MLIRTypes.i32], MLIRTypes.i32)
            Operation = Transform("compose_right", ["function_composition"])
            Namespace = "Alloy.Operators"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Operators.op_ComposeLeft"
            ShortName = "<<"
            ParameterTypes = [
                MLIRTypes.func([MLIRTypes.i32], MLIRTypes.i32)
                MLIRTypes.func([MLIRTypes.i32], MLIRTypes.i32)
            ]
            ReturnType = MLIRTypes.func([MLIRTypes.i32], MLIRTypes.i32)
            Operation = Transform("compose_left", ["function_composition"])
            Namespace = "Alloy.Operators"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        // Arithmetic operators (will delegate to Numerics functions)
        {
            QualifiedName = "Alloy.Operators.op_Addition"
            ShortName = "+"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            Operation = Transform("add_operator", ["delegates_to_add"])
            Namespace = "Alloy.Operators"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Operators.op_Subtraction"
            ShortName = "-"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            Operation = Transform("subtract_operator", ["delegates_to_subtract"])
            Namespace = "Alloy.Operators"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Operators.op_Multiply"
            ShortName = "*"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            Operation = Transform("multiply_operator", ["delegates_to_multiply"])
            Namespace = "Alloy.Operators"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Operators.op_Division"
            ShortName = "/"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            Operation = Transform("divide_operator", ["delegates_to_divide"])
            Namespace = "Alloy.Operators"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Operators.op_Modulus"
            ShortName = "%"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            Operation = Transform("modulo_operator", ["delegates_to_modulo"])
            Namespace = "Alloy.Operators"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Operators.op_Exponentiation"
            ShortName = "**"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i32
            Operation = Transform("power_operator", ["delegates_to_power"])
            Namespace = "Alloy.Operators"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        // Comparison operators
        {
            QualifiedName = "Alloy.Operators.op_Equality"
            ShortName = "="
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i1
            Operation = Transform("equals_operator", ["delegates_to_equals"])
            Namespace = "Alloy.Operators"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Operators.op_Inequality"
            ShortName = "<>"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i1
            Operation = Transform("not_equals_operator", ["delegates_to_not_equals"])
            Namespace = "Alloy.Operators"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Operators.op_LessThan"
            ShortName = "<"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i1
            Operation = Transform("less_than_operator", ["delegates_to_lessThan"])
            Namespace = "Alloy.Operators"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Operators.op_GreaterThan"
            ShortName = ">"
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i1
            Operation = Transform("greater_than_operator", ["delegates_to_greaterThan"])
            Namespace = "Alloy.Operators"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Operators.op_LessThanOrEqual"
            ShortName = "<="
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i1
            Operation = Transform("less_than_or_equal_operator", ["delegates_to_lessThanOrEqual"])
            Namespace = "Alloy.Operators"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Operators.op_GreaterThanOrEqual"
            ShortName = ">="
            ParameterTypes = [MLIRTypes.i32; MLIRTypes.i32]
            ReturnType = MLIRTypes.i1
            Operation = Transform("greater_than_or_equal_operator", ["delegates_to_greaterThanOrEqual"])
            Namespace = "Alloy.Operators"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
    ]
    
    // Memory management symbols
    let createMemorySymbols() : ResolvedSymbol list = [
        {
            QualifiedName = "Alloy.Memory.stackBuffer"
            ShortName = "stackBuffer"
            ParameterTypes = [MLIRTypes.i32]
            ReturnType = MLIRTypes.memref MLIRTypes.i8
            Operation = DialectOp(MemRef, "memref.alloca", Map["element_type", "i8"])
            Namespace = "Alloy.Memory"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Memory.spanToString"
            ShortName = "spanToString"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.memref MLIRTypes.i8
            Operation = Transform("span_to_string", ["utf8_conversion"])
            Namespace = "Alloy.Memory"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Memory.BufferOps.AsSpan"
            ShortName = "AsSpan"
            ParameterTypes = [
                MLIRTypes.memref MLIRTypes.i8  // buffer
                MLIRTypes.i32                   // offset
                MLIRTypes.i32                   // length
            ]
            ReturnType = MLIRTypes.memref MLIRTypes.i8
            Operation = Transform("buffer_slice", ["offset", "length"])
            Namespace = "Alloy.Memory.BufferOps"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
    ]
    
    // Span operations symbols
    let createSpanSymbols() : ResolvedSymbol list = [
        {
            QualifiedName = "Alloy.Span.asSpan"
            ShortName = "asSpan"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.memref MLIRTypes.i8
            Operation = Transform("array_to_span", [])
            Namespace = "Alloy.Span"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Span.sliceSpan"
            ShortName = "sliceSpan"
            ParameterTypes = [
                MLIRTypes.memref MLIRTypes.i8
                MLIRTypes.i32  // start
                MLIRTypes.i32  // length
            ]
            ReturnType = MLIRTypes.memref MLIRTypes.i8
            Operation = Transform("span_slice", ["bounds_check"])
            Namespace = "Alloy.Span"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Span.length"
            ShortName = "length"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.i32
            Operation = Transform("span_length", [])
            Namespace = "Alloy.Span"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Span.copyTo"
            ShortName = "copyTo"
            ParameterTypes = [
                MLIRTypes.memref MLIRTypes.i8  // source
                MLIRTypes.memref MLIRTypes.i8  // dest
            ]
            ReturnType = MLIRTypes.void_
            Operation = Transform("span_copy", ["memcpy_equivalent"])
            Namespace = "Alloy.Span"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Span.fill"
            ShortName = "fill"
            ParameterTypes = [
                MLIRTypes.memref MLIRTypes.i8
                MLIRTypes.i8  // value
            ]
            ReturnType = MLIRTypes.void_
            Operation = Transform("span_fill", ["memset_equivalent"])
            Namespace = "Alloy.Span"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
    ]
    
    // Console I/O symbols
    let createConsoleSymbols() : ResolvedSymbol list = [
        {
            QualifiedName = "Alloy.IO.Console.prompt"
            ShortName = "prompt"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.void_
            Operation = ExternalCall("printf", Some "libc")
            Namespace = "Alloy.IO.Console"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
        {
            QualifiedName = "Alloy.IO.Console.readInto"
            ShortName = "readInto"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.struct_([MLIRTypes.i32; MLIRTypes.i32])  // Result<int, ErrorCode>
            Operation = Composite [
                ExternalCall("fgets", Some "libc")
                ExternalCall("strlen", Some "libc")
                Transform("result_wrapper", ["success_check"])
            ]
            Namespace = "Alloy.IO.Console"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
        {
            QualifiedName = "Alloy.IO.Console.writeLine"
            ShortName = "writeLine"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.void_
            Operation = Composite [
                ExternalCall("puts", Some "libc")
            ]
            Namespace = "Alloy.IO.Console"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
    ]
    
    // String operations symbols
    let createStringSymbols() : ResolvedSymbol list = [
        {
            QualifiedName = "Alloy.IO.String.format"
            ShortName = "format"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8; MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.memref MLIRTypes.i8
            Operation = Composite [
                ExternalCall("sprintf", Some "libc")
                Transform("utf8_conversion", [])
            ]
            Namespace = "Alloy.IO.String"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
        {
            QualifiedName = "Alloy.IO.String.concat"
            ShortName = "concat"
            ParameterTypes = [
                MLIRTypes.memref MLIRTypes.i8
                MLIRTypes.memref MLIRTypes.i8
            ]
            ReturnType = MLIRTypes.memref MLIRTypes.i8
            Operation = ExternalCall("strcat", Some "libc")
            Namespace = "Alloy.IO.String"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
        {
            QualifiedName = "Alloy.IO.String.length"
            ShortName = "length"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.i32
            Operation = ExternalCall("strlen", Some "libc")
            Namespace = "Alloy.IO.String"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
        {
            QualifiedName = "Alloy.IO.String.compare"
            ShortName = "compare"
            ParameterTypes = [
                MLIRTypes.memref MLIRTypes.i8
                MLIRTypes.memref MLIRTypes.i8
            ]
            ReturnType = MLIRTypes.i32
            Operation = ExternalCall("strcmp", Some "libc")
            Namespace = "Alloy.IO.String"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
        {
            QualifiedName = "Alloy.IO.String.substring"
            ShortName = "substring"
            ParameterTypes = [
                MLIRTypes.memref MLIRTypes.i8  // source
                MLIRTypes.i32                   // start
                MLIRTypes.i32                   // length
                MLIRTypes.memref MLIRTypes.i8  // dest
            ]
            ReturnType = MLIRTypes.memref MLIRTypes.i8
            Operation = Transform("string_substring", ["bounds_check", "null_terminate"])
            Namespace = "Alloy.IO.String"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
    ]
    
    // File I/O symbols
    let createFileSymbols() : ResolvedSymbol list = [
        {
            QualifiedName = "Alloy.IO.File.exists"
            ShortName = "exists"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.i1
            Operation = ExternalCall("access", Some "libc")
            Namespace = "Alloy.IO.File"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
        {
            QualifiedName = "Alloy.IO.File.readAllBytes"
            ShortName = "readAllBytes"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]  // filename
            ReturnType = MLIRTypes.struct_([MLIRTypes.memref MLIRTypes.i8; MLIRTypes.i32])  // Result<buffer, ErrorCode>
            Operation = Composite [
                ExternalCall("fopen", Some "libc")
                ExternalCall("fseek", Some "libc")
                ExternalCall("ftell", Some "libc")
                ExternalCall("fread", Some "libc")
                ExternalCall("fclose", Some "libc")
                Transform("result_wrapper", ["file_io_check"])
            ]
            Namespace = "Alloy.IO.File"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
        {
            QualifiedName = "Alloy.IO.File.writeAllBytes"
            ShortName = "writeAllBytes"
            ParameterTypes = [
                MLIRTypes.memref MLIRTypes.i8  // filename
                MLIRTypes.memref MLIRTypes.i8  // buffer
                MLIRTypes.i32                   // length
            ]
            ReturnType = MLIRTypes.struct_([MLIRTypes.void_; MLIRTypes.i32])  // Result<unit, ErrorCode>
            Operation = Composite [
                ExternalCall("fopen", Some "libc")
                ExternalCall("fwrite", Some "libc")
                ExternalCall("fclose", Some "libc")
                Transform("result_wrapper", ["file_io_check"])
            ]
            Namespace = "Alloy.IO.File"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
    ]
    
    // Time operations symbols
    let createTimeSymbols() : ResolvedSymbol list = [
        {
            QualifiedName = "Alloy.Time.currentTicks"
            ShortName = "currentTicks"
            ParameterTypes = []
            ReturnType = MLIRTypes.i64
            Operation = Transform("platform_ticks", ["platform_specific"])
            Namespace = "Alloy.Time"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Time.currentUnixTimestamp"
            ShortName = "currentUnixTimestamp"
            ParameterTypes = []
            ReturnType = MLIRTypes.i64
            Operation = ExternalCall("time", Some "libc")
            Namespace = "Alloy.Time"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
    ]
    
    // Result type operations
    let createResultSymbols() : ResolvedSymbol list = [
        {
            QualifiedName = "Alloy.Core.Result.Ok"
            ShortName = "Ok"
            ParameterTypes = [MLIRTypes.i32]  // Value - will be resolved based on usage
            ReturnType = MLIRTypes.struct_([MLIRTypes.i32; MLIRTypes.i32])  // tag + value
            Operation = Transform("result_ok", ["tag_value"])
            Namespace = "Alloy.Core.Result"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Core.Result.Error"
            ShortName = "Error"
            ParameterTypes = [MLIRTypes.i32]  // Error - will be resolved based on usage
            ReturnType = MLIRTypes.struct_([MLIRTypes.i32; MLIRTypes.i32])  // tag + error
            Operation = Transform("result_error", ["tag_value"])
            Namespace = "Alloy.Core.Result"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
    ]

/// Registry construction and management
module RegistryConstruction =
    let buildAlloyRegistry() : CompilerResult<SymbolRegistry> =
        try
            let initialState = SymbolResolution.createEmptyState()
            
            // Gather ALL Alloy symbols
            let alloySymbols = 
                AlloySymbols.createCoreSymbols() @
                AlloySymbols.createNumericSymbols() @
                AlloySymbols.createOperatorSymbols() @
                AlloySymbols.createMemorySymbols() @
                AlloySymbols.createSpanSymbols() @
                AlloySymbols.createConsoleSymbols() @
                AlloySymbols.createStringSymbols() @
                AlloySymbols.createFileSymbols() @
                AlloySymbols.createTimeSymbols() @
                AlloySymbols.createResultSymbols()
            
            // Register all symbols
            let finalState = 
                (initialState, alloySymbols)
                ||> List.fold (fun state symbol -> SymbolResolution.registerSymbol symbol state)
            
            // Set active namespaces for Alloy imports
            let stateWithNamespaces = {
                finalState with 
                    ActiveNamespaces = [
                        "Alloy.Core"
                        "Alloy.Numerics"
                        "Alloy.Operators"
                        "Alloy.Memory"
                        "Alloy.Memory.BufferOps"
                        "Alloy.Span"
                        "Alloy.IO.Console"
                        "Alloy.IO.String"
                        "Alloy.IO.File"
                        "Alloy.Time"
                        "Alloy.Core.Result"
                    ]
            }
            
            Success {
                State = stateWithNamespaces
                ResolutionHistory = [("initialization", "Complete Alloy registry built")]
            }
            
        with ex ->
            CompilerFailure [InternalError(
                "registry construction",
                "Failed to build Alloy registry",
                Some ex.Message
            )]

/// MLIR generation helpers
module MLIRGeneration =
    let generateMLIROperation (symbol: ResolvedSymbol) (args: string list) (resultId: string) : string list =
        match symbol.Operation with
        | DialectOp(dialect, operation, attributes) ->
            let dialectPrefix = dialectToString dialect
            let attrStr = 
                if attributes.IsEmpty then ""
                else attributes |> Map.toList |> List.map (fun (k, v) -> sprintf "%s = %s" k v) |> String.concat ", " |> sprintf " {%s}"
            let argStr = if args.IsEmpty then "" else sprintf "(%s)" (String.concat ", " args)
            [sprintf "    %s = %s.%s%s : %s%s" resultId dialectPrefix operation argStr (mlirTypeToString symbol.ReturnType) attrStr]
            
        | ExternalCall(funcName, lib) ->
            let argTypes = symbol.ParameterTypes |> List.map mlirTypeToString |> String.concat ", "
            let retType = mlirTypeToString symbol.ReturnType
            [sprintf "    %s = llvm.call @%s(%s) : (%s) -> %s" resultId funcName (String.concat ", " args) argTypes retType]
            
        | Composite operations ->
            operations |> List.mapi (fun i op ->
                let stepId = sprintf "%s_step%d" resultId i
                match op with
                | ExternalCall(f, _) -> sprintf "    %s = llvm.call @%s(%s)" stepId f (String.concat ", " args)
                | _ -> sprintf "    %s = arith.constant 0 : i32  // Composite step" stepId
            )
            
        | Transform(name, parameters) ->
            match name with
            | "add_resolved" | "add_operator" -> [sprintf "    %s = arith.addi %s, %s : i32" resultId args.[0] args.[1]]
            | "subtract_resolved" | "subtract_operator" -> [sprintf "    %s = arith.subi %s, %s : i32" resultId args.[0] args.[1]]
            | "multiply_resolved" | "multiply_operator" -> [sprintf "    %s = arith.muli %s, %s : i32" resultId args.[0] args.[1]]
            | "divide_resolved" | "divide_operator" -> [sprintf "    %s = arith.divi_signed %s, %s : i32" resultId args.[0] args.[1]]
            | "modulo_resolved" | "modulo_operator" -> [sprintf "    %s = arith.remsi %s, %s : i32" resultId args.[0] args.[1]]
            | "equals_resolved" | "equals_operator" -> [sprintf "    %s = arith.cmpi eq, %s, %s : i32" resultId args.[0] args.[1]]
            | "less_than_resolved" | "less_than_operator" -> [sprintf "    %s = arith.cmpi slt, %s, %s : i32" resultId args.[0] args.[1]]
            | "greater_than_resolved" | "greater_than_operator" -> [sprintf "    %s = arith.cmpi sgt, %s, %s : i32" resultId args.[0] args.[1]]
            | "pipe_right" -> [sprintf "    %s = llvm.call %s(%s)" resultId args.[1] args.[0]]
            | _ -> [sprintf "    %s = arith.constant 0 : i32  // Transform: %s" resultId name]

/// Public interface
module PublicInterface =
    let createStandardRegistry() : CompilerResult<SymbolRegistry> =
        RegistryConstruction.buildAlloyRegistry()
    
    let resolveSymbol (name: string) (registry: SymbolRegistry) : CompilerResult<ResolvedSymbol> =
        SymbolResolution.resolveSymbol name registry.State