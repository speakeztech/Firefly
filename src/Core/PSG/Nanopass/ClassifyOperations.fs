/// Nanopass: ClassifyOperations
/// Walks App nodes and classifies them by setting the Operation field.
/// This moves all pattern recognition from the emitter into the PSG layer.
module Core.PSG.Nanopass.ClassifyOperations

open FSharp.Compiler.Symbols
open Core.PSG.Types

// ═══════════════════════════════════════════════════════════════════
// Symbol Classification Helpers
// ═══════════════════════════════════════════════════════════════════

/// Get the full qualified name from a symbol, handling various cases
let private getFullName (symbol: FSharpSymbol) : string =
    match symbol with
    | :? FSharpMemberOrFunctionOrValue as mfv ->
        try mfv.FullName with _ -> mfv.DisplayName
    | :? FSharpEntity as entity ->
        entity.TryFullName |> Option.defaultValue entity.DisplayName
    | _ -> symbol.FullName

/// Get display name from symbol
let private getDisplayName (symbol: FSharpSymbol) : string =
    symbol.DisplayName

/// Check if symbol is from a specific module
let private isFromModule (moduleName: string) (symbol: FSharpSymbol) : bool =
    let fullName = getFullName symbol
    fullName.StartsWith(moduleName + ".") || fullName = moduleName

// ═══════════════════════════════════════════════════════════════════
// Operator Classification (from SyntaxKind op_ patterns)
// ═══════════════════════════════════════════════════════════════════

/// Classify arithmetic operators from op_ syntax
let private classifyArithmeticOp (opName: string) : ArithmeticOp option =
    match opName with
    | "op_Addition" -> Some Add
    | "op_Subtraction" -> Some Sub
    | "op_Multiply" -> Some Mul
    | "op_Division" -> Some Div
    | "op_Modulus" -> Some Mod
    | "op_UnaryNegation" -> Some Negate
    | _ -> None

/// Classify bitwise operators from op_ syntax
let private classifyBitwiseOp (opName: string) : BitwiseOp option =
    match opName with
    | "op_BitwiseAnd" -> Some BitwiseAnd
    | "op_BitwiseOr" -> Some BitwiseOr
    | "op_ExclusiveOr" -> Some BitwiseXor
    | "op_LeftShift" -> Some ShiftLeft
    | "op_RightShift" -> Some ShiftRight
    | "op_LogicalNot" | "op_OnesComplement" -> Some BitwiseNot
    | _ -> None

/// Classify comparison operators from op_ syntax
let private classifyComparisonOp (opName: string) : ComparisonOp option =
    match opName with
    | "op_Equality" -> Some Eq
    | "op_Inequality" -> Some Neq
    | "op_LessThan" -> Some Lt
    | "op_GreaterThan" -> Some Gt
    | "op_LessThanOrEqual" -> Some Lte
    | "op_GreaterThanOrEqual" -> Some Gte
    | _ -> None

/// Classify operator from SyntaxKind (e.g., "LongIdent:op_Addition")
let private classifyOperator (syntaxKind: string) : OperationKind option =
    // Extract op_ name from SyntaxKind
    let opName =
        if syntaxKind.Contains("op_") then
            let idx = syntaxKind.IndexOf("op_")
            syntaxKind.Substring(idx)
        else
            ""

    if opName = "" then None
    else
        // Try arithmetic first
        match classifyArithmeticOp opName with
        | Some op -> Some (Arithmetic op)
        | None ->
            // Try bitwise
            match classifyBitwiseOp opName with
            | Some op -> Some (Bitwise op)
            | None ->
                // Try comparison
                match classifyComparisonOp opName with
                | Some op -> Some (Comparison op)
                | None -> None

// ═══════════════════════════════════════════════════════════════════
// Console Operations Classification
// ═══════════════════════════════════════════════════════════════════

let private classifyConsoleOp (displayName: string) : ConsoleOp option =
    match displayName with
    | "writeBytes" -> Some ConsoleWriteBytes
    | "readBytes" -> Some ConsoleReadBytes
    | "write" | "Write" -> Some ConsoleWrite
    | "writeln" | "writeLine" | "WriteLine" -> Some ConsoleWriteln
    | "readln" | "readLine" | "ReadLine" -> Some ConsoleReadLine
    | "readInto" | "ReadInto" | "readLineInto" -> Some ConsoleReadInto
    | "newLine" -> Some ConsoleNewLine
    | _ -> None

// ═══════════════════════════════════════════════════════════════════
// NativeString Operations Classification
// ═══════════════════════════════════════════════════════════════════

let private classifyNativeStrOp (displayName: string) : NativeStrOp option =
    match displayName with
    | "create" -> Some StrCreate
    | "empty" -> Some StrEmpty
    | "isEmpty" -> Some StrIsEmpty
    | "length" -> Some StrLength
    | "byteAt" -> Some StrByteAt
    | "copyTo" -> Some StrCopyTo
    | "ofBytes" -> Some StrOfBytes
    | "copyToBuffer" -> Some StrCopyToBuffer
    | "concat2" -> Some StrConcat2
    | "concat3" -> Some StrConcat3
    | "fromBytesTo" -> Some StrFromBytesTo
    | _ -> None

// ═══════════════════════════════════════════════════════════════════
// Memory Operations Classification
// ═══════════════════════════════════════════════════════════════════

let private classifyMemoryOp (displayName: string) : MemoryOp option =
    match displayName with
    | "stackBuffer" -> Some MemStackBuffer
    | "copy" -> Some MemCopy
    | "zero" -> Some MemZero
    | "compare" -> Some MemCompare
    | _ -> None

// ═══════════════════════════════════════════════════════════════════
// NativePtr Operations Classification
// ═══════════════════════════════════════════════════════════════════

let private classifyNativePtrOp (displayName: string) : NativePtrOp option =
    match displayName with
    | "read" -> Some PtrRead
    | "write" -> Some PtrWrite
    | "get" -> Some PtrGet
    | "set" -> Some PtrSet
    | "add" -> Some PtrAdd
    | "stackalloc" -> Some PtrStackAlloc
    | "nullPtr" -> Some PtrNull
    | "toNativeInt" -> Some PtrToNativeInt
    | "ofNativeInt" -> Some PtrOfNativeInt
    | "toVoidPtr" -> Some PtrToVoidPtr
    | "ofVoidPtr" -> Some PtrOfVoidPtr
    | _ -> None

// ═══════════════════════════════════════════════════════════════════
// Time Operations Classification
// ═══════════════════════════════════════════════════════════════════

let private classifyTimeOp (displayName: string) : TimeOp option =
    match displayName with
    | "currentTicks" -> Some CurrentTicks
    | "highResolutionTicks" -> Some HighResolutionTicks
    | "tickFrequency" -> Some TickFrequency
    | "sleep" -> Some Sleep
    | _ -> None

// ═══════════════════════════════════════════════════════════════════
// Result Operations Classification
// ═══════════════════════════════════════════════════════════════════

let private classifyResultOp (displayName: string) (fullName: string) : ResultOp option =
    match displayName with
    | "Ok" -> Some ResultOk
    | "Error" -> Some ResultError
    | _ when fullName.Contains("Result.Ok") -> Some ResultOk
    | _ when fullName.Contains("Result.Error") -> Some ResultError
    | _ -> None

// ═══════════════════════════════════════════════════════════════════
// Core Operations Classification
// ═══════════════════════════════════════════════════════════════════

let private classifyCoreOp (displayName: string) : CoreOp option =
    match displayName with
    | "ignore" -> Some Ignore
    | "failwith" -> Some Failwith
    | "invalidArg" -> Some InvalidArg
    | "not" -> Some Not
    | _ -> None

// ═══════════════════════════════════════════════════════════════════
// Conversion Operations Classification
// ═══════════════════════════════════════════════════════════════════

let private classifyConversionOp (displayName: string) : ConversionOp option =
    match displayName with
    | "byte" -> Some ToByte
    | "sbyte" -> Some ToSByte
    | "int16" -> Some ToInt16
    | "uint16" -> Some ToUInt16
    | "int" | "int32" -> Some ToInt32
    | "uint32" -> Some ToUInt32
    | "int64" -> Some ToInt64
    | "uint64" -> Some ToUInt64
    | "float32" | "single" -> Some ToFloat32
    | "float" | "double" -> Some ToFloat64
    | "char" -> Some ToChar
    | "nativeint" -> Some ToNativeInt
    | "unativeint" -> Some ToUNativeInt
    | _ -> None

// ═══════════════════════════════════════════════════════════════════
// Main Classification Logic
// ═══════════════════════════════════════════════════════════════════

/// Classify an App node based on its symbol and children
let private classifyAppNode (psg: ProgramSemanticGraph) (node: PSGNode) : OperationKind option =
    // First, check SyntaxKind for operator patterns
    if node.SyntaxKind.Contains("op_") then
        classifyOperator node.SyntaxKind
    else
        // Get the function node (first child of App)
        let children =
            match node.Children with
            | Parent ids -> ids |> List.choose (fun id -> Map.tryFind id.Value psg.Nodes)
            | _ -> []

        match children with
        | [] -> None
        | funcNode :: args ->
            // Check the function node's symbol or SyntaxKind
            let funcSymbol = funcNode.Symbol
            let funcKind = funcNode.SyntaxKind

            // Debug: print when we see Console functions
            if funcKind.Contains("Console") then
                let symInfo = match funcSymbol with
                              | Some s -> try sprintf "full=%s display=%s" (getFullName s) (getDisplayName s) with _ -> "error getting name"
                              | None -> "no symbol"
                printfn "[CLASSIFY DEBUG] Console function: kind=%s symbol=%s" funcKind symInfo

            // Check for operator in function node
            if funcKind.Contains("op_") then
                classifyOperator funcKind
            else
                // Classify based on symbol
                match funcSymbol with
                | Some symbol ->
                    let fullName = getFullName symbol
                    let displayName = getDisplayName symbol

                    // Console operations
                    if fullName.StartsWith("Alloy.Console.") then
                        classifyConsoleOp displayName |> Option.map Console
                    // NativeString operations
                    elif fullName.StartsWith("Alloy.NativeTypes.NativeString.") ||
                         fullName.StartsWith("Alloy.NativeTypes.NativeStr.") then
                        classifyNativeStrOp displayName |> Option.map NativeStr
                    // Memory operations
                    elif fullName.StartsWith("Alloy.Memory.") then
                        classifyMemoryOp displayName |> Option.map Memory
                    // NativePtr operations
                    elif fullName.Contains("NativePtr.") || fullName.Contains("NativeInterop.") then
                        classifyNativePtrOp displayName |> Option.map NativePtr
                    // Time operations
                    elif fullName.StartsWith("Alloy.Time.") then
                        classifyTimeOp displayName |> Option.map Time
                    // Result operations
                    elif fullName.Contains("Result") || displayName = "Ok" || displayName = "Error" then
                        classifyResultOp displayName fullName |> Option.map Result
                    // Core operations
                    elif fullName.StartsWith("Alloy.Core.") ||
                         fullName.StartsWith("Microsoft.FSharp.Core.Operators.") then
                        // Try core ops first
                        match classifyCoreOp displayName with
                        | Some op -> Some (Core op)
                        | None ->
                            // Try conversion ops
                            match classifyConversionOp displayName with
                            | Some op -> Some (Conversion op)
                            | None -> None
                    // Conversion operations (standalone)
                    elif classifyConversionOp displayName |> Option.isSome then
                        classifyConversionOp displayName |> Option.map Conversion
                    else
                        // Regular function call
                        let modulePath =
                            let parts = fullName.Split('.')
                            if parts.Length > 1 then
                                Some (String.concat "." parts.[0..parts.Length-2])
                            else
                                None
                        Some (RegularCall {
                            FunctionName = displayName
                            ModulePath = modulePath
                            ArgumentCount = args.Length
                        })
                | None ->
                    // No symbol - check SyntaxKind for clues
                    if funcKind.StartsWith("LongIdent:") then
                        let name = funcKind.Substring(10) // Remove "LongIdent:"
                        // Try to classify by name
                        match classifyCoreOp name with
                        | Some op -> Some (Core op)
                        | None ->
                            match classifyConversionOp name with
                            | Some op -> Some (Conversion op)
                            | None ->
                                Some (RegularCall {
                                    FunctionName = name
                                    ModulePath = None
                                    ArgumentCount = args.Length
                                })
                    else
                        None

/// Classify a single node (only processes App nodes)
let private classifyNode (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode =
    if node.SyntaxKind.StartsWith("App") && node.Operation.IsNone then
        let result = classifyAppNode psg node
        // Debug Console operations
        if node.SyntaxKind.Contains("Console") || (node.Symbol |> Option.exists (fun s -> s.FullName.Contains("Console"))) then
            printfn "[CLASSIFY NODE] App node: %s result=%A" node.SyntaxKind result
        match result with
        | Some op -> { node with Operation = Some op }
        | None -> node
    else
        node

// ═══════════════════════════════════════════════════════════════════
// Nanopass Entry Point
// ═══════════════════════════════════════════════════════════════════

/// Classify all App nodes in the PSG
let classifyOperations (psg: ProgramSemanticGraph) : ProgramSemanticGraph =
    printfn "[NANOPASS] ClassifyOperations: Classifying App nodes..."

    let mutable classified = 0
    let mutable unclassified = 0

    let newNodes =
        psg.Nodes
        |> Map.map (fun _ node ->
            let result = classifyNode psg node
            if result.Operation.IsSome && node.Operation.IsNone then
                classified <- classified + 1
            elif node.SyntaxKind.StartsWith("App") && result.Operation.IsNone then
                unclassified <- unclassified + 1
            result)

    printfn "[NANOPASS] ClassifyOperations: Classified %d App nodes, %d remain unclassified" classified unclassified

    { psg with Nodes = newNodes }
