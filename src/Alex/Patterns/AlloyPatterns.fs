/// AlloyPatterns - Predicates and classifiers for Alloy library operations
///
/// DESIGN PRINCIPLE:
/// XParsec (via PSGXParsec) is THE combinator infrastructure.
/// This module provides only:
/// - Predicates: PSGNode -> bool (for use with PSGXParsec.satisfyChild)
/// - Classifiers: PSGNode -> 'T option (for extracting operation types)
/// - Helper functions to identify Alloy library calls
///
/// NO custom combinators here. Use XParsec for composition.
///
/// ARCHITECTURE NOTE:
/// Function calls in the PSG have this structure:
///   App:FunctionCall [ModuleName]
///   +-- LongIdent:Module.Function [FunctionName]  <- The actual function symbol
///   +-- Arg1, Arg2, ...                           <- Arguments
///
/// So classifiers must look at the node's children or symbol to identify operations.
module Alex.Patterns.AlloyPatterns

open Core.PSG.Types
open FSharp.Compiler.Symbols

// ===================================================================
// Safe Symbol Helpers
// ===================================================================

/// Safely get a symbol's FullName, handling types like 'unit' that throw exceptions
let private tryGetFullName (sym: FSharpSymbol) : string option =
    try Some sym.FullName with _ -> None

/// Get a symbol's FullName or a fallback value
let private getFullNameOrDefault (sym: FSharpSymbol) (fallback: string) : string =
    match tryGetFullName sym with
    | Some name -> name
    | None -> fallback

// ===================================================================
// Type Definitions
// ===================================================================

/// Console operation types
type ConsoleOp =
    | Write
    | WriteLine
    | ReadLine
    | Read

/// Time operation types
type TimeOp =
    | CurrentTicks
    | HighResolutionTicks
    | TickFrequency
    | Sleep
    | CurrentUnixTimestamp
    | CurrentTimestamp
    | CurrentDateTimeString

/// Text.Format operation types
type TextFormatOp =
    | IntToString
    | Int64ToString
    | FloatToString
    | BoolToString

/// Memory operation types
type MemoryOp =
    | StackBuffer of elementType: string option  // stackBuffer<T>
    | SpanToString  // Convert Span<byte> to string
    | AsReadOnlySpan  // Get read-only span from buffer
    // Semantic memory primitives - express WHAT not HOW
    | Copy           // Memory.copy src dest len
    | CopyElements   // Memory.copyElements<T> src dest count
    | Zero           // Memory.zero dest len
    | ZeroElements   // Memory.zeroElements<T> dest count
    | Fill           // Memory.fill dest value len
    | Compare        // Memory.compare a b len

/// NativeStr operation types (from Alloy.NativeTypes.NativeString)
type NativeStrOp =
    | StrCreate      // NativeStr(ptr, len) constructor
    | StrEmpty       // empty() -> NativeStr
    | StrIsEmpty     // isEmpty s -> bool
    | StrLength      // length s -> int
    | StrByteAt      // byteAt index s -> byte
    | StrCopyTo      // copyTo dest s -> int
    // Semantic string primitives
    | StrOfBytes     // ofBytes bytes -> NativeStr (from byte literal)
    | StrCopyToBuffer// copyToBuffer dest s -> NativeStr
    | StrConcat2     // concat2 dest a b -> NativeStr
    | StrConcat3     // concat3 dest a b c -> NativeStr
    | StrFromBytesTo // fromBytesTo dest bytes -> NativeStr

/// Result DU constructor types
type ResultOp =
    | OkCtor    // Ok value - constructs Result with tag 0
    | ErrorCtor // Error value - constructs Result with tag 1

/// NativePtr operation types (from FSharp.NativeInterop or Alloy.NativeTypes)
type NativePtrOp =
    | PtrGet    // NativePtr.get ptr index -> value
    | PtrSet    // NativePtr.set ptr index value -> unit
    | PtrRead   // NativePtr.read ptr -> value
    | PtrWrite  // NativePtr.write ptr value -> unit
    | PtrAdd    // NativePtr.add ptr offset -> ptr
    | PtrToInt  // NativePtr.toNativeInt ptr -> nativeint
    | PtrNull   // NativePtr.nullPtr<T> -> null pointer
    | PtrToVoid // NativePtr.toVoidPtr ptr -> voidptr (just a cast to ptr)
    | PtrOfVoid // NativePtr.ofVoidPtr ptr -> nativeptr (just a cast from voidptr)

/// Core operation types (from Alloy.Core)
type CoreOp =
    | Ignore  // ignore value -> unit (discards value, returns unit)

/// All recognized Alloy operations
type AlloyOp =
    | Console of ConsoleOp
    | Time of TimeOp
    | TextFormat of TextFormatOp
    | Memory of MemoryOp
    | NativeStr of NativeStrOp
    | Result of ResultOp
    | Core of CoreOp

/// Arithmetic operation types
type ArithOp =
    | Add | Sub | Mul | Div | Mod
    | BitwiseAnd | BitwiseOr | BitwiseXor
    | ShiftLeft | ShiftRight
    | Negate | Not

/// Type conversion operation types (F# core operators)
type ConversionOp =
    | ToByte    // byte x - truncate to i8
    | ToSByte   // sbyte x - truncate to i8 (signed)
    | ToInt16   // int16 x - extend/truncate to i16
    | ToUInt16  // uint16 x - extend/truncate to i16
    | ToInt32   // int x / int32 x - extend/truncate to i32
    | ToUInt32  // uint32 x - extend/truncate to i32
    | ToInt64   // int64 x - extend to i64
    | ToUInt64  // uint64 x - extend to i64
    | ToFloat   // float32 x - convert to f32
    | ToDouble  // float x - convert to f64

/// Comparison operation types
type CompareOp =
    | Eq | Neq | Lt | Gt | Lte | Gte

/// Information about a function call
/// Uses PSG node for type information - no FCS lookups after PSG construction
type FunctionCallInfo = {
    AppNode: PSGNode             // The App node (has result Type field)
    FunctionNode: PSGNode        // The function identifier node
    FunctionName: string         // Display name for MLIR emission
    Arguments: PSGNode list      // Argument nodes
}

/// Information about a pipe expression (value |> func)
type PipeInfo = {
    Value: PSGNode       // The left side of the pipe (the value being piped)
    Function: PSGNode    // The right side of the pipe (the function to apply)
}

// ===================================================================
// Helper: Get children of a PSG node
// ===================================================================

/// Get child nodes from a PSGNode using its Children state
/// Note: Children are in correct source order after PSG finalization
let private getChildNodes (graph: ProgramSemanticGraph) (node: PSGNode) : PSGNode list =
    match node.Children with
    | ChildrenState.Parent ids ->
        ids |> List.choose (fun id -> Map.tryFind id.Value graph.Nodes)
    | _ -> []

/// Get the function symbol from an App node's children
/// Handles PSG child ordering by looking for the child that is most likely the function reference:
/// - LongIdent or Ident nodes (direct function references)
/// - For curried apps, we may need to recurse into inner Apps
let rec private getFunctionSymbolFromChildren (graph: ProgramSemanticGraph) (children: PSGNode list) : FSharpSymbol option =
    // First, try to find a LongIdent (qualified function reference like Console.readLine)
    children
    |> List.tryFind (fun c -> c.SyntaxKind.StartsWith("LongIdent:"))
    |> Option.bind (fun c -> c.Symbol)
    |> Option.orElseWith (fun () ->
        // Then try to find an Ident with a function type
        children
        |> List.tryFind (fun c ->
            c.SyntaxKind.StartsWith("Ident:") &&
            match c.Type with
            | Some ftype -> ftype.IsFunctionType
            | None -> false)
        |> Option.bind (fun c -> c.Symbol))
    |> Option.orElseWith (fun () ->
        // Then recurse into App children to find the function
        children
        |> List.tryPick (fun c ->
            if c.SyntaxKind.StartsWith("App") then
                let innerChildren = getChildNodes graph c
                getFunctionSymbolFromChildren graph innerChildren
            else None))

// ===================================================================
// Symbol Classification Helpers
// ===================================================================

/// Classify a Console operation from a symbol
let classifyConsoleOp (symbol: FSharpSymbol) : ConsoleOp option =
    let fullName = getFullNameOrDefault symbol ""
    let displayName = symbol.DisplayName

    // Check for WriteLine (must check before Write since Write is substring)
    if displayName = "WriteLine" ||
       fullName.EndsWith(".WriteLine") ||
       fullName.Contains("Console.WriteLine") then
        Some WriteLine
    // Check for Write
    elif displayName = "Write" ||
         fullName.EndsWith(".Write") ||
         fullName.Contains("Console.Write") then
        Some Write
    // Check for Prompt (same as Write semantically)
    elif displayName = "Prompt" ||
         fullName.EndsWith(".Prompt") ||
         fullName.Contains("Console.Prompt") then
        Some Write  // Prompt behaves like Write
    // Check for ReadLine (public API only, not internal readLine function)
    // Console.ReadLine is the user-facing API that gets inlined
    // Console.readLine (lowercase) is the internal implementation that should be a normal function call
    elif displayName = "ReadLine" ||
         fullName.Contains("Console.ReadLine") then
        Some ReadLine
    // Note: readInto is NOT special-cased here - it uses SRTP which resolves
    // to concrete trait member calls. The TraitCall nodes are emitted properly
    // and resolve to the actual member implementations.
    else
        None

/// Classify a Time operation from a symbol
let classifyTimeOp (symbol: FSharpSymbol) : TimeOp option =
    let fullName = getFullNameOrDefault symbol ""
    let displayName = symbol.DisplayName

    if displayName = "currentTicks" || fullName.Contains("currentTicks") then
        Some CurrentTicks
    elif displayName = "highResolutionTicks" || fullName.Contains("highResolutionTicks") then
        Some HighResolutionTicks
    elif displayName = "tickFrequency" || fullName.Contains("tickFrequency") then
        Some TickFrequency
    elif displayName = "sleep" || displayName = "Sleep" || fullName.Contains("sleep") then
        Some Sleep
    elif displayName = "currentUnixTimestamp" || fullName.Contains("currentUnixTimestamp") then
        Some CurrentUnixTimestamp
    elif displayName = "currentTimestamp" || fullName.Contains("currentTimestamp") then
        Some CurrentTimestamp
    elif displayName = "currentDateTimeString" || fullName.Contains("currentDateTimeString") then
        Some CurrentDateTimeString
    else
        None

/// Classify a Text.Format operation from a symbol
let classifyTextFormatOp (symbol: FSharpSymbol) : TextFormatOp option =
    let fullName = getFullNameOrDefault symbol ""
    let displayName = symbol.DisplayName

    if displayName = "intToString" || fullName.Contains("intToString") then
        Some IntToString
    elif displayName = "int64ToString" || fullName.Contains("int64ToString") then
        Some Int64ToString
    elif displayName = "floatToString" || fullName.Contains("floatToString") then
        Some FloatToString
    elif displayName = "boolToString" || fullName.Contains("boolToString") then
        Some BoolToString
    else
        None

/// Classify a Memory operation from a symbol
let classifyMemoryOp (symbol: FSharpSymbol) : MemoryOp option =
    let fullName = getFullNameOrDefault symbol ""
    let displayName = symbol.DisplayName

    if displayName = "stackBuffer" || fullName.Contains("Memory.stackBuffer") then
        Some (StackBuffer None)  // Type parameter extracted separately if needed
    elif displayName = "spanToString" || fullName.Contains("spanToString") then
        Some SpanToString
    elif displayName = "AsReadOnlySpan" || fullName.Contains("AsReadOnlySpan") then
        Some AsReadOnlySpan
    // Semantic memory primitives
    elif displayName = "copy" && fullName.Contains("Memory") then
        Some Copy
    elif displayName = "copyElements" && fullName.Contains("Memory") then
        Some CopyElements
    elif displayName = "zero" && fullName.Contains("Memory") then
        Some Zero
    elif displayName = "zeroElements" && fullName.Contains("Memory") then
        Some ZeroElements
    elif displayName = "fill" && fullName.Contains("Memory") then
        Some Fill
    elif displayName = "compare" && fullName.Contains("Memory") then
        Some Compare
    else
        None

/// Classify a NativeStr operation from a symbol
let classifyNativeStrOp (symbol: FSharpSymbol) : NativeStrOp option =
    let fullName = getFullNameOrDefault symbol ""
    let displayName = symbol.DisplayName

    // Check for NativeString module functions
    let isNativeString = fullName.Contains("NativeString") || fullName.Contains("NativeStr")

    if not isNativeString then None
    elif displayName = ".ctor" || displayName = "NativeStr" then
        Some StrCreate
    elif displayName = "empty" then
        Some StrEmpty
    elif displayName = "isEmpty" then
        Some StrIsEmpty
    elif displayName = "length" then
        Some StrLength
    elif displayName = "byteAt" then
        Some StrByteAt
    elif displayName = "copyTo" then
        Some StrCopyTo
    // Semantic string primitives
    elif displayName = "ofBytes" then
        Some StrOfBytes
    elif displayName = "copyToBuffer" then
        Some StrCopyToBuffer
    elif displayName = "concat2" then
        Some StrConcat2
    elif displayName = "concat3" then
        Some StrConcat3
    elif displayName = "fromBytesTo" then
        Some StrFromBytesTo
    else
        None

/// Classify a Result DU constructor operation from a symbol
let classifyResultOp (symbol: FSharpSymbol) : ResultOp option =
    let fullName = getFullNameOrDefault symbol ""
    let displayName = symbol.DisplayName

    // Match Ok and Error DU constructors
    // They may appear as Alloy.Core.Ok, Alloy.Core.Error, or just Ok/Error
    if displayName = "Ok" || fullName.Contains(".Ok") || fullName.EndsWith(".Ok") then
        Some OkCtor
    elif displayName = "Error" || fullName.Contains(".Error") || fullName.EndsWith(".Error") then
        Some ErrorCtor
    else
        None

/// Classify a NativePtr operation from a symbol
let classifyNativePtrOp (symbol: FSharpSymbol) : NativePtrOp option =
    let fullName = getFullNameOrDefault symbol ""
    let displayName = symbol.DisplayName

    // Match NativePtr operations from FSharp.NativeInterop or Alloy.NativeTypes
    if displayName = "get" && (fullName.Contains("NativePtr") || fullName.Contains("NativeInterop")) then
        Some PtrGet
    elif displayName = "set" && (fullName.Contains("NativePtr") || fullName.Contains("NativeInterop")) then
        Some PtrSet
    elif displayName = "read" && (fullName.Contains("NativePtr") || fullName.Contains("NativeInterop")) then
        Some PtrRead
    elif displayName = "write" && (fullName.Contains("NativePtr") || fullName.Contains("NativeInterop")) then
        Some PtrWrite
    elif displayName = "add" && (fullName.Contains("NativePtr") || fullName.Contains("NativeInterop")) then
        Some PtrAdd
    elif displayName = "toNativeInt" && (fullName.Contains("NativePtr") || fullName.Contains("NativeInterop")) then
        Some PtrToInt
    elif displayName = "nullPtr" && (fullName.Contains("NativePtr") || fullName.Contains("NativeInterop")) then
        Some PtrNull
    elif displayName = "toVoidPtr" && (fullName.Contains("NativePtr") || fullName.Contains("NativeInterop")) then
        Some PtrToVoid
    elif displayName = "ofVoidPtr" && (fullName.Contains("NativePtr") || fullName.Contains("NativeInterop")) then
        Some PtrOfVoid
    else
        None

/// Classify an arithmetic operator from an MFV
let classifyArithOp (mfv: FSharpMemberOrFunctionOrValue) : ArithOp option =
    match mfv.CompiledName with
    | "op_Addition" -> Some Add
    | "op_Subtraction" -> Some Sub
    | "op_Multiply" -> Some Mul
    | "op_Division" -> Some Div
    | "op_Modulus" -> Some Mod
    | "op_BitwiseAnd" -> Some BitwiseAnd
    | "op_BitwiseOr" -> Some BitwiseOr
    | "op_ExclusiveOr" -> Some BitwiseXor
    | "op_LeftShift" -> Some ShiftLeft
    | "op_RightShift" -> Some ShiftRight
    | "op_UnaryNegation" -> Some Negate
    | "op_LogicalNot" -> Some Not
    | "not" -> Some Not  // CompiledName for F# not function
    | _ when mfv.DisplayName = "not" || mfv.DisplayName = "``not``" -> Some Not
    | _ -> None

/// Classify a comparison operator from an MFV
let classifyCompareOp (mfv: FSharpMemberOrFunctionOrValue) : CompareOp option =
    match mfv.CompiledName with
    | "op_Equality" -> Some Eq
    | "op_Inequality" -> Some Neq
    | "op_LessThan" -> Some Lt
    | "op_GreaterThan" -> Some Gt
    | "op_LessThanOrEqual" -> Some Lte
    | "op_GreaterThanOrEqual" -> Some Gte
    | _ -> None

/// Classify a type conversion operator from an MFV
let classifyConversionOp (mfv: FSharpMemberOrFunctionOrValue) : ConversionOp option =
    let displayName = mfv.DisplayName
    let fullName = getFullNameOrDefault mfv ""
    // Match F# core conversion operators
    if displayName = "byte" || fullName.EndsWith(".byte") || fullName.Contains("Operators.byte") then
        Some ToByte
    elif displayName = "sbyte" || fullName.EndsWith(".sbyte") || fullName.Contains("Operators.sbyte") then
        Some ToSByte
    elif displayName = "int16" || fullName.EndsWith(".int16") || fullName.Contains("Operators.int16") then
        Some ToInt16
    elif displayName = "uint16" || fullName.EndsWith(".uint16") || fullName.Contains("Operators.uint16") then
        Some ToUInt16
    elif displayName = "int" || displayName = "int32" || fullName.EndsWith(".int") || fullName.EndsWith(".int32") ||
         fullName.Contains("Operators.int") || fullName.Contains("Operators.int32") then
        Some ToInt32
    elif displayName = "uint32" || fullName.EndsWith(".uint32") || fullName.Contains("Operators.uint32") then
        Some ToUInt32
    elif displayName = "int64" || fullName.EndsWith(".int64") || fullName.Contains("Operators.int64") then
        Some ToInt64
    elif displayName = "uint64" || fullName.EndsWith(".uint64") || fullName.Contains("Operators.uint64") then
        Some ToUInt64
    elif displayName = "float32" || fullName.EndsWith(".float32") || fullName.Contains("Operators.float32") then
        Some ToFloat
    elif displayName = "float" || displayName = "double" || fullName.EndsWith(".float") ||
         fullName.Contains("Operators.float") then
        Some ToDouble
    else
        None

// ===================================================================
// Node Classifiers - Extract operation type from PSGNode
// ===================================================================

/// Extract Console operation from a node (checks children recursively for function symbol)
let extractConsoleOp (graph: ProgramSemanticGraph) (node: PSGNode) : ConsoleOp option =
    if not (node.SyntaxKind.StartsWith("App")) then None
    else
        let children = getChildNodes graph node
        match getFunctionSymbolFromChildren graph children with
        | Some symbol -> classifyConsoleOp symbol
        | None ->
            // Also check the node's own symbol
            match node.Symbol with
            | Some s -> classifyConsoleOp s
            | None -> None

/// Extract Time operation from a node
let extractTimeOp (graph: ProgramSemanticGraph) (node: PSGNode) : TimeOp option =
    if not (node.SyntaxKind.StartsWith("App")) then None
    else
        let children = getChildNodes graph node
        match getFunctionSymbolFromChildren graph children with
        | Some symbol -> classifyTimeOp symbol
        | None ->
            match node.Symbol with
            | Some s -> classifyTimeOp s
            | None -> None

/// Extract TextFormat operation from a node
let extractTextFormatOp (graph: ProgramSemanticGraph) (node: PSGNode) : TextFormatOp option =
    if not (node.SyntaxKind.StartsWith("App")) then None
    else
        let children = getChildNodes graph node
        match getFunctionSymbolFromChildren graph children with
        | Some symbol -> classifyTextFormatOp symbol
        | None ->
            match node.Symbol with
            | Some s -> classifyTextFormatOp s
            | None -> None

/// Extract Memory operation from a node
let extractMemoryOp (graph: ProgramSemanticGraph) (node: PSGNode) : MemoryOp option =
    if not (node.SyntaxKind.StartsWith("App")) then None
    else
        let children = getChildNodes graph node
        match getFunctionSymbolFromChildren graph children with
        | Some symbol -> classifyMemoryOp symbol
        | None ->
            match node.Symbol with
            | Some s -> classifyMemoryOp s
            | None -> None

/// Extract NativeStr operation from a node
let extractNativeStrOp (graph: ProgramSemanticGraph) (node: PSGNode) : NativeStrOp option =
    if not (node.SyntaxKind.StartsWith("App")) then None
    else
        let children = getChildNodes graph node
        match getFunctionSymbolFromChildren graph children with
        | Some symbol -> classifyNativeStrOp symbol
        | None ->
            match node.Symbol with
            | Some s -> classifyNativeStrOp s
            | None -> None

/// Extract Result DU constructor operation from a node
let extractResultOp (graph: ProgramSemanticGraph) (node: PSGNode) : ResultOp option =
    if not (node.SyntaxKind.StartsWith("App")) then None
    else
        let children = getChildNodes graph node
        match getFunctionSymbolFromChildren graph children with
        | Some symbol -> classifyResultOp symbol
        | None ->
            match node.Symbol with
            | Some s -> classifyResultOp s
            | None -> None

/// Extract NativePtr operation from a node
/// NativePtr operations are curried: NativePtr.set ptr idx value is ((NativePtr.set ptr) idx) value
/// Also handles TypeApp nodes for generic operations like nullPtr<byte>
let extractNativePtrOp (graph: ProgramSemanticGraph) (node: PSGNode) : NativePtrOp option =
    // Handle TypeApp nodes (like nullPtr<byte>) - check child for NativePtr symbol
    if node.SyntaxKind.StartsWith("TypeApp") then
        let children = getChildNodes graph node
        match getFunctionSymbolFromChildren graph children with
        | Some symbol -> classifyNativePtrOp symbol
        | None ->
            // Also check all children for LongIdent with NativePtr
            children
            |> List.tryPick (fun child ->
                match child.Symbol with
                | Some sym -> classifyNativePtrOp sym
                | None -> None)
    elif not (node.SyntaxKind.StartsWith("App")) then None
    else
        let children = getChildNodes graph node
        // Look through curried applications to find the NativePtr function
        let rec findPtrOp (children: PSGNode list) : NativePtrOp option =
            match getFunctionSymbolFromChildren graph children with
            | Some symbol ->
                match classifyNativePtrOp symbol with
                | Some op -> Some op
                | None ->
                    // Check if first child is another App (curried call)
                    match children with
                    | innerApp :: _ when innerApp.SyntaxKind.StartsWith("App") ->
                        findPtrOp (getChildNodes graph innerApp)
                    | _ -> None
            | None ->
                match children with
                | innerApp :: _ when innerApp.SyntaxKind.StartsWith("App") ->
                    findPtrOp (getChildNodes graph innerApp)
                | _ -> None

        findPtrOp children

/// Extract type conversion operation from a node
/// Conversion ops are unary: byte value is (byte) value
let extractConversionOp (graph: ProgramSemanticGraph) (node: PSGNode) : ConversionOp option =
    if not (node.SyntaxKind.StartsWith("App")) then None
    else
        let children = getChildNodes graph node
        match getFunctionSymbolFromChildren graph children with
        | Some symbol ->
            match symbol with
            | :? FSharpMemberOrFunctionOrValue as mfv -> classifyConversionOp mfv
            | _ -> None
        | None ->
            // Also check the node's own symbol
            match node.Symbol with
            | Some (:? FSharpMemberOrFunctionOrValue as mfv) -> classifyConversionOp mfv
            | _ -> None

/// Classify a symbol as a Core operation
let private classifyCoreOp (sym: FSharpSymbol) : CoreOp option =
    match sym with
    | :? FSharpMemberOrFunctionOrValue as mfv ->
        let displayName = mfv.DisplayName
        let fullName = getFullNameOrDefault sym displayName
        // Check for Alloy.Core.ignore
        if displayName = "ignore" && (fullName.Contains("Alloy") || fullName.Contains("Core")) then
            Some Ignore
        else
            None
    | _ -> None

/// Extract Core operation from a node
/// Core ops like 'ignore' are simple unary functions: ignore x
let extractCoreOp (graph: ProgramSemanticGraph) (node: PSGNode) : CoreOp option =
    if not (node.SyntaxKind.StartsWith("App")) then None
    else
        let children = getChildNodes graph node
        match getFunctionSymbolFromChildren graph children with
        | Some symbol -> classifyCoreOp symbol
        | None -> None

/// Extract any Alloy operation from a node
let extractAlloyOp (graph: ProgramSemanticGraph) (node: PSGNode) : AlloyOp option =
    match extractConsoleOp graph node with
    | Some op -> Some (Console op)
    | None ->
    match extractTimeOp graph node with
    | Some op -> Some (Time op)
    | None ->
    match extractTextFormatOp graph node with
    | Some op -> Some (TextFormat op)
    | None ->
    match extractMemoryOp graph node with
    | Some op -> Some (Memory op)
    | None ->
    match extractNativeStrOp graph node with
    | Some op -> Some (NativeStr op)
    | None ->
    match extractResultOp graph node with
    | Some op -> Some (Result op)
    | None ->
    match extractCoreOp graph node with
    | Some op -> Some (Core op)
    | None -> None

/// Check if node is a NativeStr operation
let isNativeStrCall (graph: ProgramSemanticGraph) (node: PSGNode) : bool =
    extractNativeStrOp graph node |> Option.isSome

/// Extract arithmetic operation from an App node
/// Handles curried operators: a + b is ((+) a) b
/// For curried form, the structure is: App[App[op, leftArg], rightArg]
/// The operator is the FIRST child of the FIRST App child
/// IMPORTANT: Only matches COMPLETE binary applications with both operands.
/// Partial applications like ((+) a) are NOT matched - they need to be
/// emitted as partial function applications.
let extractArithOp (graph: ProgramSemanticGraph) (node: PSGNode) : ArithOp option =
    if not (node.SyntaxKind.StartsWith("App")) then None
    else
        let children = getChildNodes graph node

        // Helper to check if a node is an arithmetic operator
        let isArithOpNode (opNode: PSGNode) =
            let result =
                opNode.SyntaxKind.StartsWith("LongIdent:op_") ||
                opNode.SyntaxKind.StartsWith("Ident:op_") ||
                opNode.SyntaxKind = "Ident:not" ||
                opNode.SyntaxKind.Contains(":not")
            result

        // Find the inner App among children (PSG order may be [arg, App] or [App, arg])
        let findInnerApp (children: PSGNode list) =
            children |> List.tryFind (fun c -> c.SyntaxKind.StartsWith("App"))

        // Find an operator node in children
        let findOpNode (children: PSGNode list) =
            children |> List.tryFind isArithOpNode

        // For binary operators, we need:
        // 1. Two children in the outer App (inner partial app + right operand)
        // 2. The inner App must contain the operator + left operand
        match children with
        | [c1; c2] ->
            // Two children - could be a complete binary operation
            match findInnerApp children with
            | Some innerApp ->
                // The outer App has an inner App and the right operand
                // Now check if inner App contains an operator
                let innerChildren = getChildNodes graph innerApp
                match findOpNode innerChildren with
                | Some opNode ->
                    match opNode.Symbol with
                    | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
                        classifyArithOp mfv
                    | _ -> None
                | None -> None
            | None ->
                // No inner App - might be a unary operator like "not x"
                // For unary ops, we have [opNode, operandNode] directly
                match findOpNode children with
                | Some opNode ->
                    match opNode.Symbol with
                    | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
                        // Only unary operators (Negate, Not) should match here
                        match classifyArithOp mfv with
                        | Some Negate -> Some Negate
                        | Some Not -> Some Not
                        | _ -> None  // Binary operators need the curried form
                    | _ -> None
                | None -> None
        | [_] ->
            // Single child - might be unary operator application (rare case)
            None
        | _ ->
            // More than 2 children or empty - not a valid operator application
            None

/// Extract comparison operation from an App node
/// Handles curried operators: a >= b is ((>=) a) b
/// For curried form, the structure is: App[App[op, leftArg], rightArg]
/// PSG children order may be [arg, App] or [App, arg]
/// IMPORTANT: Only matches COMPLETE binary comparisons with both operands.
/// Partial applications are NOT matched.
let extractCompareOp (graph: ProgramSemanticGraph) (node: PSGNode) : CompareOp option =
    if not (node.SyntaxKind.StartsWith("App")) then None
    else
        let children = getChildNodes graph node

        // Helper to check if a node is a comparison operator
        let isCompareOpNode (opNode: PSGNode) =
            opNode.SyntaxKind.StartsWith("LongIdent:op_") ||
            opNode.SyntaxKind.StartsWith("Ident:op_")

        // Find the inner App among children
        let findInnerApp (children: PSGNode list) =
            children |> List.tryFind (fun c -> c.SyntaxKind.StartsWith("App"))

        // Find a comparison operator node in children
        let findOpNode (children: PSGNode list) =
            children |> List.tryFind isCompareOpNode

        // For binary comparison operators, we need:
        // 1. Two children in the outer App (inner partial app + right operand)
        // 2. The inner App must contain the operator + left operand
        match children with
        | [_; _] ->
            // Two children - could be a complete comparison
            match findInnerApp children with
            | Some innerApp ->
                // The outer App has an inner App and the right operand
                let innerChildren = getChildNodes graph innerApp
                match findOpNode innerChildren with
                | Some opNode ->
                    match opNode.Symbol with
                    | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
                        classifyCompareOp mfv
                    | _ -> None
                | None -> None
            | None ->
                // No inner App - not a valid curried comparison
                None
        | _ ->
            // Not two children - not a valid comparison operator application
            None

// ===================================================================
// Node Predicates - Boolean checks for PSGNode
// ===================================================================

/// Check if node is a Console operation
let isConsoleCall (graph: ProgramSemanticGraph) (node: PSGNode) : bool =
    extractConsoleOp graph node |> Option.isSome

/// Check if node is a Time operation
let isTimeCall (graph: ProgramSemanticGraph) (node: PSGNode) : bool =
    extractTimeOp graph node |> Option.isSome

/// Check if node is a TextFormat operation
let isTextFormatCall (graph: ProgramSemanticGraph) (node: PSGNode) : bool =
    extractTextFormatOp graph node |> Option.isSome

/// Check if node is a Memory operation
let isMemoryCall (graph: ProgramSemanticGraph) (node: PSGNode) : bool =
    extractMemoryOp graph node |> Option.isSome

/// Check if node is any Alloy operation
let isAlloyCall (graph: ProgramSemanticGraph) (node: PSGNode) : bool =
    extractAlloyOp graph node |> Option.isSome

/// Check if node is an arithmetic operation
let isArithCall (graph: ProgramSemanticGraph) (node: PSGNode) : bool =
    extractArithOp graph node |> Option.isSome

/// Check if node is a comparison operation
let isCompareCall (graph: ProgramSemanticGraph) (node: PSGNode) : bool =
    extractCompareOp graph node |> Option.isSome

// ===================================================================
// Pipe Operator Extraction
// ===================================================================

/// Check if a symbol is the pipe operator (|>)
let private isPipeOperator (sym: FSharpSymbol) : bool =
    match sym with
    | :? FSharpMemberOrFunctionOrValue as mfv ->
        mfv.DisplayName = "op_PipeRight" || mfv.CompiledName = "op_PipeRight"
    | _ -> false

/// Extract pipe expression information from an App node
/// The pipe operator (|>) has structure: App[App[op_PipeRight, value], func]
/// Where value |> func means func(value)
let extractPipe (graph: ProgramSemanticGraph) (node: PSGNode) : PipeInfo option =
    if not (node.SyntaxKind.StartsWith("App")) then None
    else
        let children = getChildNodes graph node
        match children with
        | [innerApp; funcNode] when innerApp.SyntaxKind.StartsWith("App") ->
            // Check if innerApp contains the pipe operator
            let innerChildren = getChildNodes graph innerApp
            match innerChildren with
            | [pipeOpNode; valueNode] ->
                match pipeOpNode.Symbol with
                | Some sym when isPipeOperator sym ->
                    Some { Value = valueNode; Function = funcNode }
                | _ -> None
            | _ -> None
        | _ -> None

/// Check if node is a pipe expression
let isPipe (graph: ProgramSemanticGraph) (node: PSGNode) : bool =
    extractPipe graph node |> Option.isSome

// ===================================================================
// Function Call Info Extraction
// ===================================================================

/// Extract function call information from an App node
/// Handles curried calls by recursively traversing nested App nodes
/// For `f a b c`, the PSG structure is: App(App(App(f, a), b), c)
/// PSG children order may be [arg, func] due to construction order
/// This function collects all arguments from all curry levels
/// Uses PSG node data only - no FCS lookups
let extractFunctionCall (graph: ProgramSemanticGraph) (node: PSGNode) : FunctionCallInfo option =
    if not (node.SyntaxKind.StartsWith("App")) then None
    else
        // Helper to check if a node is a function/partial application (not a plain value)
        let isFuncOrPartial (n: PSGNode) =
            n.SyntaxKind.StartsWith("App") ||
            n.SyntaxKind.StartsWith("LongIdent:") ||
            n.SyntaxKind.StartsWith("Ident:") ||
            n.SyntaxKind.StartsWith("TypeApp:")

        // Helper to check if a node looks like a value/argument (not a function ref)
        let isLikelyArg (n: PSGNode) =
            n.SyntaxKind.StartsWith("Const") ||
            n.SyntaxKind.StartsWith("PropertyAccess") ||
            n.SyntaxKind.StartsWith("AddressOf") ||
            n.SyntaxKind.StartsWith("DotIndexedGet")

        // Helper to check if a node's type is a function type (has arrows)
        let hasFunctionType (n: PSGNode) : bool =
            match n.Type with
            | Some ftype -> ftype.IsFunctionType
            | None -> false

        // Given two children, determine which is the func/partial and which is the arg
        let classifyChildren (c1: PSGNode) (c2: PSGNode) : (PSGNode * PSGNode) =
            // If one is clearly a value (Const, etc), it's the arg
            if isLikelyArg c1 && not (isLikelyArg c2) then (c2, c1)
            elif isLikelyArg c2 && not (isLikelyArg c1) then (c1, c2)
            // If one is a LongIdent/Ident (function ref) and the other is App,
            // the App is the partial application, LongIdent is the function
            elif c1.SyntaxKind.StartsWith("LongIdent:") && c2.SyntaxKind.StartsWith("App") then (c2, c1)
            elif c2.SyntaxKind.StartsWith("LongIdent:") && c1.SyntaxKind.StartsWith("App") then (c1, c2)
            elif c1.SyntaxKind.StartsWith("Ident:") && c2.SyntaxKind.StartsWith("App") then (c2, c1)
            elif c2.SyntaxKind.StartsWith("Ident:") && c1.SyntaxKind.StartsWith("App") then (c1, c2)
            // Both are Apps - first one is usually the arg in PSG structure
            elif c1.SyntaxKind.StartsWith("App") && c2.SyntaxKind.StartsWith("App") then (c2, c1)
            // Both are Idents - use type information to determine which is the function
            // The one with a function type (arrows) is the function, the other is the arg
            elif c1.SyntaxKind.StartsWith("Ident:") && c2.SyntaxKind.StartsWith("Ident:") then
                if hasFunctionType c2 && not (hasFunctionType c1) then (c2, c1)
                elif hasFunctionType c1 && not (hasFunctionType c2) then (c1, c2)
                else (c2, c1)  // Default if both or neither have function types
            // If c2 is a function reference (LongIdent/Ident), c1 is the arg
            elif c2.SyntaxKind.StartsWith("LongIdent:") || c2.SyntaxKind.StartsWith("Ident:") then (c2, c1)
            elif c1.SyntaxKind.StartsWith("LongIdent:") || c1.SyntaxKind.StartsWith("Ident:") then (c1, c2)
            // Default: assume c1 is func, c2 is arg (original assumption)
            else (c1, c2)

        // Helper to check if an App is an operator application (should not recurse into)
        let isOperatorApp (appNode: PSGNode) =
            let appChildren = getChildNodes graph appNode
            appChildren |> List.exists (fun c ->
                c.SyntaxKind.StartsWith("LongIdent:op_") ||
                c.SyntaxKind.StartsWith("Ident:op_") ||
                c.SyntaxKind.StartsWith("LongIdent:not") ||
                c.SyntaxKind = "Ident:not")

        // Recursively traverse nested App nodes to find the function and collect all arguments
        // STOPS at operator applications (arithmetic, comparison) - those are values, not partial applications
        // Returns None if the final "function" is actually an operator application
        let rec collectCurriedArgs (appNode: PSGNode) (accArgs: PSGNode list) : (PSGNode * PSGNode list) option =
            let children = getChildNodes graph appNode
            match children with
            | [c1; c2] ->
                let (funcOrApp, arg) = classifyChildren c1 c2
                let newArgs = arg :: accArgs
                // Only recurse if it's an App AND not an operator application
                if funcOrApp.SyntaxKind.StartsWith("App") then
                    if isOperatorApp funcOrApp then
                        // The "function" is actually an operator application - this whole expression
                        // is an arithmetic/comparison operation, not a function call
                        None
                    else
                        collectCurriedArgs funcOrApp newArgs
                else
                    Some (funcOrApp, newArgs)
            | [single] ->
                if single.SyntaxKind.StartsWith("App") then
                    if isOperatorApp single then
                        None  // Operator application, not a function call
                    else
                        collectCurriedArgs single accArgs
                else
                    Some (single, accArgs)
            | first :: rest ->
                // More than 2 children - use classification on first two
                let (funcOrApp, arg) = classifyChildren first (List.head rest)
                let remainingArgs = List.tail rest
                let newArgs = arg :: remainingArgs @ accArgs
                if funcOrApp.SyntaxKind.StartsWith("App") then
                    if isOperatorApp funcOrApp then
                        None  // Operator application, not a function call
                    else
                        collectCurriedArgs funcOrApp newArgs
                else
                    Some (funcOrApp, newArgs)
            | [] -> None

        match collectCurriedArgs node [] with
        | Some (funcNode, allArgs) ->
            match funcNode.Symbol with
            | Some sym ->
                Some {
                    AppNode = node          // The outermost App node has the final result type
                    FunctionNode = funcNode
                    FunctionName = getFullNameOrDefault sym sym.DisplayName
                    Arguments = allArgs
                }
            | None ->
                // Function node doesn't have a symbol directly, but might be a LongIdent or TypeApp
                // Try to get the name from the syntax kind
                let name =
                    if funcNode.SyntaxKind.StartsWith("LongIdent:") then
                        funcNode.SyntaxKind.Substring(10)
                    elif funcNode.SyntaxKind.StartsWith("Ident:") then
                        funcNode.SyntaxKind.Substring(6)
                    else
                        funcNode.SyntaxKind
                if name <> "" && name <> funcNode.SyntaxKind then
                    Some {
                        AppNode = node
                        FunctionNode = funcNode
                        FunctionName = name
                        Arguments = allArgs
                    }
                else None
        | None -> None

// ===================================================================
// MLIR Operation Names for Operators
// ===================================================================

/// Get MLIR operation name for arithmetic op
let arithOpToMLIR (op: ArithOp) : string =
    match op with
    | Add -> "arith.addi"
    | Sub -> "arith.subi"
    | Mul -> "arith.muli"
    | Div -> "arith.divsi"
    | Mod -> "arith.remsi"
    | BitwiseAnd -> "arith.andi"
    | BitwiseOr -> "arith.ori"
    | BitwiseXor -> "arith.xori"
    | ShiftLeft -> "arith.shli"
    | ShiftRight -> "arith.shrsi"
    | Negate -> "arith.subi" // Will need special handling (0 - x)
    | Not -> "arith.xori"    // Will need special handling (xor with -1)

/// Get MLIR predicate for comparison op
let compareOpToPredicate (op: CompareOp) : string =
    match op with
    | Eq -> "eq"
    | Neq -> "ne"
    | Lt -> "slt"
    | Gt -> "sgt"
    | Lte -> "sle"
    | Gte -> "sge"
