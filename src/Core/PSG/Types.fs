module Core.PSG.Types

open System
open FSharp.Compiler.Text
open FSharp.Compiler.Symbols

/// Unique identifier for nodes in the PSG
type NodeId = {
    Value: string
}
with
    static member Create(value: string) = { Value = value }
    static member FromRange(fileName: string, range: range) =
        let cleanFileName = System.IO.Path.GetFileNameWithoutExtension(fileName)
        let rangeStr = sprintf "%d_%d_%d_%d" range.Start.Line range.Start.Column range.End.Line range.End.Column
        { Value = sprintf "rng_%s_%s" cleanFileName rangeStr }
    static member FromSymbol(symbol: FSharpSymbol) =
        let hashCode = symbol.GetHashCode().ToString("x8")
        { Value = sprintf "sym_%s_%s" symbol.DisplayName hashCode }

/// Child processing state with compile-time guarantees
type ChildrenState =
    | NotProcessed
    | Parent of NodeId list
    | NoChildren

/// Types of edges in the PSG
type EdgeKind =
    | ChildOf
    | FunctionCall
    | SymRef
    | TypeOf
    | Instantiates
    | SymbolDef
    | SymbolUse
    | TypeInstantiation of typeArgs: FSharpType list
    | ControlFlow of kind: string
    | DataDependency
    | ModuleContainment

/// Edge in the program semantic graph
type PSGEdge = {
    Source: NodeId
    Target: NodeId
    Kind: EdgeKind
}

/// Symbol relationship types for analysis
type SymbolRelation =
    | DefinesType of FSharpEntity
    | UsesType of FSharpEntity
    | CallsSymbol of FSharpMemberOrFunctionOrValue
    | ImplementsInterface of FSharpEntity
    | InheritsFrom of FSharpEntity
    | ReferencesSymbol of FSharpSymbol

/// Context requirements for continuation compilation decisions
type ContextRequirement =
    | Pure              // No external dependencies
    | AsyncBoundary     // Suspension point
    | ResourceAccess    // File/network access
    | Parameter of int  // Function parameter with index

/// Computation patterns for optimization decisions
type ComputationPattern =
    | DataDriven        // Push-based, eager evaluation
    | DemandDriven      // Pull-based, lazy evaluation

/// Constant values for Const nodes
/// Extracted at PSG construction time so downstream doesn't parse SyntaxKind
type ConstantValue =
    | StringValue of string
    | Int32Value of int
    | Int64Value of int64
    | FloatValue of float
    | BoolValue of bool
    | CharValue of char
    | ByteValue of byte
    | UnitValue

// ═══════════════════════════════════════════════════════════════════
// Typed Syntax Kind - Type-driven dispatch instead of string parsing
// ═══════════════════════════════════════════════════════════════════

/// Expression syntax kinds (mirrors FCS SynExpr cases)
/// Names and other data come from Symbol/Type fields, not embedded here
type ExprKind =
    | EApp           // Function application
    | ETypeApp       // Generic type application (e.g., stackalloc<byte>)
    | EIdent         // Simple identifier
    | ELongIdent     // Qualified identifier (e.g., Module.func)
    | EConst         // Constant value (actual value in ConstantValue field)
    | ELetOrUse      // Let or use binding
    | ESequential    // Sequential expressions
    | EIfThenElse    // Conditional
    | EMatch         // Pattern match
    | ELambda        // Lambda expression
    | EForLoop       // For loop
    | EWhileLoop     // While loop
    | ETryWith       // Try/with exception handling
    | ETryFinally    // Try/finally
    | EMethodCall    // Method call (obj.Method)
    | EPropertyAccess // Property access (obj.Property)
    | EMutableSet    // Mutable variable assignment
    | EAddressOf     // Address-of operator (&)
    | ERecord        // Record construction
    | ETuple         // Tuple construction
    | EArrayOrList   // Array or list construction
    | EIndexGet      // Indexer get (arr.[i])
    | EIndexSet      // Indexer set (arr.[i] <- v)
    | ETraitCall     // SRTP trait call
    | EInterpolatedString // Interpolated string literal
    | EObjExpr       // Object expression
    | EUpcast        // Upcast (:>)
    | EDowncast      // Downcast (:?>)

/// Pattern syntax kinds (mirrors FCS SynPat cases)
type PatternKind =
    | PNamed        // Named pattern (binding)
    | PLongIdent    // Union case or qualified name
    | PWild         // Wildcard (_)
    | PConst        // Constant pattern
    | PTuple        // Tuple pattern
    | PRecord       // Record pattern
    | PArrayOrList  // Array/list pattern
    | PAs           // As pattern
    | POr           // Or pattern
    | PTyped        // Typed pattern

/// Declaration syntax kinds (mirrors FCS SynModuleDecl/SynTypeDefn)
type DeclKind =
    | DModule       // Module declaration
    | DNestedModule // Nested module
    | DTypeDefn     // Type definition
    | DBinding      // Value/function binding
    | DOpen         // Open declaration
    | DAttribute    // Attribute application
    | DException    // Exception type

/// Binding kinds
type BindingKind =
    | BLet          // let x = ...
    | BUse          // use x = ...
    | BMutable      // let mutable x = ...
    | BFunction     // let f x = ...

/// Top-level syntax kind - the typed alternative to string SyntaxKind
type SyntaxKindT =
    | SKExpr of ExprKind
    | SKPattern of PatternKind
    | SKDecl of DeclKind
    | SKBinding of BindingKind
    | SKUnknown  // For cases not yet classified (temporary during migration)

module SyntaxKindT =
    /// Convert typed SyntaxKind to string (for backwards compatibility during migration)
    let toString (sk: SyntaxKindT) : string =
        match sk with
        | SKExpr e -> sprintf "Expr:%A" e
        | SKPattern p -> sprintf "Pattern:%A" p
        | SKDecl d -> sprintf "Decl:%A" d
        | SKBinding b -> sprintf "Binding:%A" b
        | SKUnknown -> "Unknown"

    /// Check if this is an expression kind
    let isExpr = function SKExpr _ -> true | _ -> false

    /// Check if this is a pattern kind
    let isPattern = function SKPattern _ -> true | _ -> false

    /// Check if this is a declaration kind
    let isDecl = function SKDecl _ -> true | _ -> false

    /// Check if this is a binding kind
    let isBinding = function SKBinding _ -> true | _ -> false

    /// Extract expression kind if present
    let tryGetExpr = function SKExpr e -> Some e | _ -> None

    /// Extract pattern kind if present
    let tryGetPattern = function SKPattern p -> Some p | _ -> None

// ═══════════════════════════════════════════════════════════════════
// Library Operation Classifications
// These types describe the semantic operation at an App node,
// enabling the emitter to dispatch directly without re-analyzing symbols.
// ═══════════════════════════════════════════════════════════════════

/// Arithmetic operations (emit as arith dialect ops)
type ArithmeticOp =
    | Add | Sub | Mul | Div | Mod
    | Negate  // Unary negation

/// Bitwise operations (emit as arith dialect ops)
type BitwiseOp =
    | BitwiseAnd | BitwiseOr | BitwiseXor
    | ShiftLeft | ShiftRight
    | BitwiseNot  // Unary complement

/// Comparison operations (emit as arith.cmpi)
type ComparisonOp =
    | Eq | Neq | Lt | Gt | Lte | Gte

/// Type conversion operations
type ConversionOp =
    | ToByte | ToSByte
    | ToInt16 | ToUInt16
    | ToInt32 | ToUInt32
    | ToInt64 | ToUInt64
    | ToFloat32 | ToFloat64
    | ToChar
    | ToNativeInt | ToUNativeInt

/// NativePtr operations (from FSharp.NativeInterop)
type NativePtrOp =
    | PtrRead       // NativePtr.read ptr
    | PtrWrite      // NativePtr.write ptr value
    | PtrGet        // NativePtr.get ptr index
    | PtrSet        // NativePtr.set ptr index value
    | PtrAdd        // NativePtr.add ptr offset
    | PtrStackAlloc // NativePtr.stackalloc<T> count
    | PtrNull       // NativePtr.nullPtr<T>
    | PtrToNativeInt
    | PtrOfNativeInt
    | PtrToVoidPtr
    | PtrOfVoidPtr

/// Console I/O operations (platform-specific emission)
type ConsoleOp =
    | ConsoleWriteBytes   // writeBytes fd ptr count
    | ConsoleReadBytes    // readBytes fd ptr count
    | ConsoleWrite        // write (high-level)
    | ConsoleWriteln      // writeln
    | ConsoleReadLine     // readLine
    | ConsoleReadInto     // readInto (SRTP)
    | ConsoleNewLine      // newLine

/// Time operations (platform-specific emission)
type TimeOp =
    | CurrentTicks
    | HighResolutionTicks
    | TickFrequency
    | Sleep

/// NativeString operations
type NativeStrOp =
    | StrCreate       // NativeStr(ptr, len) constructor
    | StrEmpty        // empty()
    | StrIsEmpty      // isEmpty s
    | StrLength       // length s
    | StrByteAt       // byteAt index s
    | StrCopyTo       // copyTo dest s
    | StrOfBytes      // ofBytes bytes
    | StrCopyToBuffer // copyToBuffer dest s
    | StrConcat2      // concat2 dest a b
    | StrConcat3      // concat3 dest a b c
    | StrConcatN      // concatN - variable number of args (>3)
    | StrFromBytesTo  // fromBytesTo dest bytes

/// Memory operations
type MemoryOp =
    | MemStackBuffer  // Memory.stackBuffer<T> size
    | MemCopy         // Memory.copy src dest len
    | MemZero         // Memory.zero dest len
    | MemCompare      // Memory.compare a b len

/// Result DU operations
type ResultOp =
    | ResultOk        // Ok value
    | ResultError     // Error err

/// Core operations
type CoreOp =
    | Ignore          // ignore value
    | Failwith        // failwith message
    | InvalidArg      // invalidArg paramName message
    | Not             // not (boolean negation)

/// Text formatting operations
type TextFormatOp =
    | IntToString
    | Int64ToString
    | FloatToString
    | BoolToString

/// Information about a regular function call (for unclassified App nodes)
type RegularCallInfo = {
    FunctionName: string
    ModulePath: string option
    ArgumentCount: int
}

/// Classified operation for App nodes
/// The nanopass ClassifyOperations sets this on App nodes
/// so the emitter can dispatch directly without re-analyzing symbols.
type OperationKind =
    | Arithmetic of ArithmeticOp
    | Bitwise of BitwiseOp
    | Comparison of ComparisonOp
    | Conversion of ConversionOp
    | NativePtr of NativePtrOp
    | Console of ConsoleOp
    | Time of TimeOp
    | NativeStr of NativeStrOp
    | Memory of MemoryOp
    | Result of ResultOp
    | Core of CoreOp
    | TextFormat of TextFormatOp
    | RegularCall of RegularCallInfo
    // Note: Pipe operators (|>, <|) are reduced by ReducePipeOperators nanopass
    // and should never appear as OperationKind

/// Represents a single overload candidate for SRTP resolution.
/// Contains all information needed to select the correct overload at emission time.
/// All type information is serialized as strings to avoid FCS type dependencies downstream.
type SRTPOverloadCandidate = {
    /// Full name of the target method (e.g., "Alloy.Console.writeSystemString")
    TargetMethodFullName: string
    /// Parameter type names for matching (e.g., ["Alloy.Console.WritableString"; "Microsoft.FSharp.Core.string"])
    ParameterTypeNames: string list
    /// Return type name
    ReturnTypeName: string
}

/// SRTP (Statically Resolved Type Parameter) resolution information.
/// Captured from FCS internal TraitConstraintInfo.Solution via reflection.
/// This represents the resolved concrete implementation of a trait constraint.
type SRTPResolution =
    /// Resolved to an F# method: (declaringType, methodRef, methodTypeInstantiation)
    | FSMethod of declaringType: FSharpType * methodRef: FSharpMemberOrFunctionOrValue * methodTypeArgs: FSharpType list
    /// Resolved to an F# method by name (from FCS internals via reflection)
    /// Used when we can extract the method name but not the full FSharpMemberOrFunctionOrValue
    | FSMethodByName of methodFullName: string
    /// Multiple overloads available - selection deferred to emission based on argument types.
    /// Contains full signature information for each candidate so emission can match without FCS.
    | MultipleOverloads of traitName: string * candidates: SRTPOverloadCandidate list
    /// Resolved to an F# record field
    | FSRecordField of fieldType: FSharpType * fieldName: string * isSetProperty: bool
    /// Resolved to an anonymous record field
    | FSAnonRecordField of recordType: FSharpType * fieldIndex: int
    /// Resolved to a built-in operator (no explicit implementation needed)
    | BuiltIn
    /// Resolution not available (generic code or reflection failure)
    | Unresolved of reason: string

/// PSG node with soft-delete support added to existing structure
type PSGNode = {
    // EXISTING FIELDS - DO NOT CHANGE
    Id: NodeId
    SyntaxKind: string  // DEPRECATED: Use Kind field for type-driven dispatch
    Symbol: FSharpSymbol option
    Type: FSharpType option
    Constraints: FSharpGenericParameterConstraint list option
    Range: range
    SourceFile: string
    ParentId: NodeId option
    Children: ChildrenState

    // NEW FIELDS - Soft-delete support
    IsReachable: bool
    EliminationPass: int option
    EliminationReason: string option
    ReachabilityDistance: int option

    // NEW FIELDS - Context tracking for continuation compilation
    ContextRequirement: ContextRequirement option
    ComputationPattern: ComputationPattern option

    // NEW FIELD - Operation classification (set by ClassifyOperations nanopass)
    Operation: OperationKind option

    // NEW FIELD - Constant value for Const nodes (set by PSG Builder)
    ConstantValue: ConstantValue option

    // NEW FIELD - Typed syntax kind for type-driven dispatch (set by PSG Builder)
    // Use this instead of parsing SyntaxKind string
    Kind: SyntaxKindT

    // NEW FIELD - SRTP resolution (set by ResolveSRTP nanopass)
    // For TraitCall expressions, this contains the resolved concrete implementation
    SRTPResolution: SRTPResolution option
}

/// Complete Program Semantic Graph
type ProgramSemanticGraph = {
    Nodes: Map<string, PSGNode>
    Edges: PSGEdge list
    SymbolTable: Map<string, FSharpSymbol>
    EntryPoints: NodeId list
    SourceFiles: Map<string, string>
    CompilationOrder: string list
    /// String literals discovered during PSG construction.
    /// Maps hash -> content for deduplication.
    /// Populated during FCS ingestion, consumed during MLIR emission.
    StringLiterals: Map<uint32, string>
}

/// Result type for PSG operations
type PSGResult<'T> = 
    | Success of 'T
    | Failure of PSGError list

and PSGError = {
    Message: string
    Location: range option
    ErrorKind: PSGErrorKind
}

and PSGErrorKind =
    | CorrelationFailure
    | MissingSymbol
    | InvalidNode
    | BuilderError
    | TypeResolutionError
    | MemoryAnalysisError

/// Helper functions for working with ChildrenState
module ChildrenStateHelpers =

    /// Create a new node with NotProcessed children state
    /// The `kind` parameter defaults to Unknown for backwards compatibility during migration
    let createWithNotProcessed id syntaxKind symbol range sourceFile parentId = {
        Id = id
        SyntaxKind = syntaxKind
        Symbol = symbol
        Type = None
        Constraints = None
        Range = range
        SourceFile = sourceFile
        ParentId = parentId
        Children = NotProcessed

        // Initialize new soft-delete fields with defaults
        IsReachable = false
        EliminationPass = None
        EliminationReason = None
        ReachabilityDistance = None

        // Initialize context tracking fields
        ContextRequirement = None
        ComputationPattern = None

        // Initialize operation classification
        Operation = None

        // Initialize constant value
        ConstantValue = None

        // Initialize typed syntax kind (Unknown during migration)
        Kind = SKUnknown

        // Initialize SRTP resolution
        SRTPResolution = None
    }

    /// Create a new node with typed syntax kind
    let createWithKind id syntaxKind kind symbol range sourceFile parentId = {
        Id = id
        SyntaxKind = syntaxKind
        Symbol = symbol
        Type = None
        Constraints = None
        Range = range
        SourceFile = sourceFile
        ParentId = parentId
        Children = NotProcessed

        // Initialize new soft-delete fields with defaults
        IsReachable = false
        EliminationPass = None
        EliminationReason = None
        ReachabilityDistance = None

        // Initialize context tracking fields
        ContextRequirement = None
        ComputationPattern = None

        // Initialize operation classification
        Operation = None

        // Initialize constant value
        ConstantValue = None

        // Set typed syntax kind
        Kind = kind

        // Initialize SRTP resolution
        SRTPResolution = None
    }

    /// Add a child to a node (appends to maintain source order)
    let addChild childId node =
        match node.Children with
        | NotProcessed -> { node with Children = Parent [childId] }
        | Parent existing -> { node with Children = Parent (existing @ [childId]) }
        | NoChildren -> { node with Children = Parent [childId] }

    /// Mark unprocessed nodes as having no children
    let finalizeChildren node =
        match node.Children with
        | NotProcessed -> { node with Children = NoChildren }
        | _ -> node

    /// Get children as list
    let getChildrenList node =
        match node.Children with
        | NotProcessed -> []
        | NoChildren -> []
        | Parent children -> children

/// Helper functions for soft-delete reachability
module ReachabilityHelpers =
    
    /// Mark a node as reachable
    let markReachable (distance: int) (node: PSGNode) =
        { node with 
            IsReachable = true
            ReachabilityDistance = Some distance
            EliminationPass = None
            EliminationReason = None }
    
    /// Mark a node as unreachable
    let markUnreachable (pass: int) (reason: string) (node: PSGNode) =
        { node with 
            IsReachable = false
            EliminationPass = Some pass
            EliminationReason = Some reason
            ReachabilityDistance = None }
    
    /// Check if node is reachable
    let isReachable (node: PSGNode) = node.IsReachable
    
    /// Analyze context requirement from syntax kind and symbol
    let analyzeContextRequirement (node: PSGNode) : ContextRequirement option =
        // First check for resource management patterns in syntax
        match node.SyntaxKind with
        | "LetOrUse:Use" | "Binding:Use" -> Some ResourceAccess
        | "TryWith" | "TryFinally" -> Some ResourceAccess
        | sk when sk.Contains("Console.") -> Some ResourceAccess  // Any Console operation is IO
        | _ ->
            // Then check symbol information for async/IO patterns
            match node.Symbol with
            | Some symbol ->
                let fullName = symbol.FullName
                // Check if this is any Console-related symbol
                if fullName.StartsWith("Alloy.Console.") || 
                   fullName = "Console" ||
                   fullName.Contains(".Write") ||
                   fullName.Contains(".Read") ||
                   fullName.Contains(".WriteLine") ||
                   fullName.Contains(".ReadLine") then
                    Some ResourceAccess
                else
                    match symbol with
                    | :? FSharpMemberOrFunctionOrValue as mfv ->
                        try
                            // Check for async computation expressions
                            let returnType = 
                                try mfv.ReturnParameter.Type
                                with _ -> failwith "Failed to get ReturnParameter.Type"
                            if returnType.HasTypeDefinition then
                                match returnType.TypeDefinition.TryFullName with
                                | Some fullName when fullName.StartsWith("Microsoft.FSharp.Control.FSharpAsync") ->
                                    Some AsyncBoundary
                                | _ ->
                                    // Check for known IO operations
                                    if mfv.FullName.StartsWith("Alloy.Console.") ||
                                       mfv.FullName.Contains("File.") ||
                                       mfv.FullName.Contains("Stream") ||
                                       mfv.FullName.Contains("Reader") ||
                                       mfv.FullName.Contains("Writer") ||
                                       mfv.DisplayName = "Write" ||
                                       mfv.DisplayName = "WriteLine" ||
                                       mfv.DisplayName = "Read" ||
                                       mfv.DisplayName = "ReadLine" then
                                        Some ResourceAccess
                                    // Check for buffer/memory operations that need cleanup
                                    elif mfv.FullName.Contains("stackBuffer") ||
                                         mfv.FullName.Contains("Buffer") ||
                                         mfv.FullName.Contains("Span") then
                                        Some ResourceAccess
                                    else
                                        Some Pure
                            else
                                // No type definition - check by name patterns
                                if mfv.FullName.StartsWith("Alloy.Console.") ||
                                   mfv.FullName.Contains("File.") ||
                                   mfv.FullName.Contains("Stream") ||
                                   mfv.FullName.Contains("Reader") ||
                                   mfv.FullName.Contains("Writer") ||
                                   mfv.DisplayName = "Write" ||
                                   mfv.DisplayName = "WriteLine" ||
                                   mfv.DisplayName = "Read" ||
                                   mfv.DisplayName = "ReadLine" then
                                    Some ResourceAccess
                                elif mfv.FullName.Contains("stackBuffer") ||
                                     mfv.FullName.Contains("Buffer") ||
                                     mfv.FullName.Contains("Span") then
                                    Some ResourceAccess
                                else
                                    Some Pure
                        with _ -> Some Pure
                    | :? FSharpEntity as entity ->
                        // Check if this is a Console module
                        if entity.FullName = "Alloy.Console" || entity.DisplayName = "Console" then
                            Some ResourceAccess
                        else
                            Some Pure
                    | _ -> Some Pure
            | None -> 
                // Fallback to syntax kind analysis
                match node.SyntaxKind with
                | sk when sk.StartsWith("Const:") -> Some Pure
                | sk when sk.Contains("Sequential") -> None  // Inherit from children
                | _ -> None
    
    /// Analyze computation pattern from node structure
    let analyzeComputationPattern (node: PSGNode) : ComputationPattern option =
        match node.Symbol with
        | Some symbol ->
            match symbol with
            | :? FSharpMemberOrFunctionOrValue as mfv ->
                try
                    let returnType = mfv.ReturnParameter.Type
                    // Check for lazy/seq types in return type
                    if returnType.HasTypeDefinition then
                        match returnType.TypeDefinition.TryFullName with
                        | Some typeDef ->
                            if typeDef.Contains("IEnumerable") || 
                               typeDef.Contains("Lazy") ||
                               typeDef.Contains("seq") ||
                               typeDef.Contains("AsyncSeq") then
                                Some DemandDriven
                            elif typeDef.Contains("FSharpList") ||
                                 typeDef.Contains("Array") ||
                                 typeDef.Contains("ResizeArray") then
                                Some DataDriven
                            else
                                // Check if function is curried (partial application = demand-driven)
                                if mfv.CurriedParameterGroups.Count > 1 then
                                    Some DemandDriven
                                else
                                    Some DataDriven
                        | None ->
                            // No qualified name (primitive type), default to data-driven
                            Some DataDriven
                    else
                        // No type definition, default to data-driven
                        Some DataDriven
                with _ -> Some DataDriven
            | _ -> Some DataDriven
        | None ->
            // Fallback to syntax kind patterns
            match node.SyntaxKind with
            | sk when sk.Contains("Sequential") -> Some DataDriven
            | sk when sk.Contains("Match") -> Some DataDriven
            | sk when sk.Contains("Lambda") -> Some DemandDriven
            | _ -> None
    
    /// Update node with analyzed context
    let updateNodeContext (node: PSGNode) =
        try
            { node with
                ContextRequirement = node.ContextRequirement |> Option.orElse (analyzeContextRequirement node)
                ComputationPattern = node.ComputationPattern |> Option.orElse (analyzeComputationPattern node) }
        with _ -> node

/// Helper functions for operation classification and MLIR emission
module OperationHelpers =

    /// Get MLIR operation name for arithmetic op
    let arithmeticOpToMLIR (op: ArithmeticOp) : string =
        match op with
        | Add -> "arith.addi"
        | Sub -> "arith.subi"
        | Mul -> "arith.muli"
        | Div -> "arith.divsi"
        | Mod -> "arith.remsi"
        | Negate -> "arith.subi" // 0 - x

    /// Get MLIR operation name for bitwise op
    let bitwiseOpToMLIR (op: BitwiseOp) : string =
        match op with
        | BitwiseAnd -> "arith.andi"
        | BitwiseOr -> "arith.ori"
        | BitwiseXor -> "arith.xori"
        | ShiftLeft -> "arith.shli"
        | ShiftRight -> "arith.shrsi"
        | BitwiseNot -> "arith.xori" // xor with -1

    /// Get MLIR predicate for comparison op (for arith.cmpi)
    let comparisonOpToPredicate (op: ComparisonOp) : string =
        match op with
        | Eq -> "eq"
        | Neq -> "ne"
        | Lt -> "slt"
        | Gt -> "sgt"
        | Lte -> "sle"
        | Gte -> "sge"

    /// Check if operation is a binary arithmetic op
    let isBinaryArithOp (op: ArithmeticOp) : bool =
        match op with
        | Add | Sub | Mul | Div | Mod -> true
        | Negate -> false

    /// Check if operation is a binary bitwise op
    let isBinaryBitwiseOp (op: BitwiseOp) : bool =
        match op with
        | BitwiseAnd | BitwiseOr | BitwiseXor | ShiftLeft | ShiftRight -> true
        | BitwiseNot -> false