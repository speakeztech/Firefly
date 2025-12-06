module Core.MLIR.Emitter

open System
open System.Text
open FSharp.Compiler.Symbols
open FSharp.Compiler.Syntax
open Core.PSG.Types
open Core.Types.MLIRTypes

/// SSA value counter for generating unique names
type SSAContext = {
    mutable Counter: int
    mutable Locals: Map<string, string>  // F# name -> SSA name
    mutable StringLiterals: (string * string) list  // content -> global name
}

module SSAContext =
    let create() = { Counter = 0; Locals = Map.empty; StringLiterals = [] }

    let nextValue ctx =
        let name = sprintf "%%v%d" ctx.Counter
        ctx.Counter <- ctx.Counter + 1
        name

    let bindLocal ctx fsharpName =
        let ssaName = nextValue ctx
        ctx.Locals <- Map.add fsharpName ssaName ctx.Locals
        ssaName

    let lookupLocal ctx name =
        Map.tryFind name ctx.Locals

    let registerLocal ctx fsharpName ssaName =
        ctx.Locals <- Map.add fsharpName ssaName ctx.Locals

    let addStringLiteral ctx content =
        match ctx.StringLiterals |> List.tryFind (fun (c, _) -> c = content) with
        | Some (_, name) -> name
        | None ->
            let name = sprintf "@str%d" (List.length ctx.StringLiterals)
            ctx.StringLiterals <- (content, name) :: ctx.StringLiterals
            name

/// MLIR operation builder
type MLIRBuilder = {
    mutable Indent: int
    Output: StringBuilder
}

module MLIRBuilder =
    let create() = { Indent = 0; Output = StringBuilder() }

    let indent b = String.replicate (b.Indent * 2) " "

    let line b text =
        b.Output.AppendLine(indent b + text) |> ignore

    let lineNoIndent b (text: string) =
        b.Output.AppendLine(text) |> ignore

    let push b = b.Indent <- b.Indent + 1
    let pop b = b.Indent <- b.Indent - 1

    let toString b = b.Output.ToString()

/// Convert F# type to MLIR type string
let rec fsharpTypeToMLIR (ftype: FSharpType) : string =
    try
        if ftype.IsAbbreviation then
            fsharpTypeToMLIR ftype.AbbreviatedType
        elif ftype.IsFunctionType then
            let args = ftype.GenericArguments
            if args.Count >= 2 then
                let paramType = fsharpTypeToMLIR args.[0]
                let retType = fsharpTypeToMLIR args.[1]
                sprintf "(%s) -> %s" paramType retType
            else "() -> i32"
        elif ftype.IsTupleType then
            let elemTypes =
                ftype.GenericArguments
                |> Seq.map fsharpTypeToMLIR
                |> String.concat ", "
            sprintf "tuple<%s>" elemTypes
        elif ftype.HasTypeDefinition then
            match ftype.TypeDefinition.TryFullName with
            | Some "System.Int32" -> "i32"
            | Some "System.Int64" -> "i64"
            | Some "System.Int16" -> "i16"
            | Some "System.Byte" -> "i8"
            | Some "System.Boolean" -> "i1"
            | Some "System.Single" -> "f32"
            | Some "System.Double" -> "f64"
            | Some "System.String" -> "!llvm.ptr"
            | Some "System.Void" | Some "Microsoft.FSharp.Core.unit" -> "()"
            | Some "System.IntPtr" | Some "System.UIntPtr" -> "!llvm.ptr"
            | Some name when name.StartsWith("Microsoft.FSharp.Core.nativeptr") -> "!llvm.ptr"
            | Some name when name.StartsWith("Microsoft.FSharp.Core.FSharpOption") ->
                let innerType =
                    if ftype.GenericArguments.Count > 0 then
                        fsharpTypeToMLIR ftype.GenericArguments.[0]
                    else "i32"
                sprintf "!variant<Some: %s, None: ()>" innerType
            | Some name when name.StartsWith("Microsoft.FSharp.Collections.FSharpList") ->
                let elemType =
                    if ftype.GenericArguments.Count > 0 then
                        fsharpTypeToMLIR ftype.GenericArguments.[0]
                    else "i32"
                sprintf "!llvm.ptr" // Lists are pointers
            | Some name when ftype.TypeDefinition.IsArrayType ->
                let elemType =
                    if ftype.GenericArguments.Count > 0 then
                        fsharpTypeToMLIR ftype.GenericArguments.[0]
                    else "i32"
                sprintf "memref<?x%s>" elemType
            // Use i32 as fallback for unknown types - proper type mapping will be added later
            | Some name -> "i32"
            | None -> "i32"
        else "i32"
    with _ -> "i32"

/// Get MLIR return type for function
let getFunctionReturnType (mfv: FSharpMemberOrFunctionOrValue) : string =
    try
        fsharpTypeToMLIR mfv.ReturnParameter.Type
    with _ -> "i32"

/// Get MLIR parameter types for function
let getFunctionParamTypes (mfv: FSharpMemberOrFunctionOrValue) : (string * string) list =
    try
        mfv.CurriedParameterGroups
        |> Seq.collect id
        |> Seq.map (fun p ->
            let name = p.DisplayName
            let mlirType = fsharpTypeToMLIR p.Type
            (name, mlirType))
        |> Seq.toList
    with _ -> []

/// Emit syscall for Linux write
let emitSyscallWrite (builder: MLIRBuilder) (ctx: SSAContext) (fd: string) (bufPtr: string) (len: string) =
    let sysWrite = "1"  // Linux x86_64 write syscall number
    let fdVal = SSAContext.nextValue ctx
    let sysNum = SSAContext.nextValue ctx
    let result = SSAContext.nextValue ctx

    MLIRBuilder.line builder (sprintf "%s = arith.constant %s : i64" fdVal fd)
    MLIRBuilder.line builder (sprintf "%s = arith.constant %s : i64" sysNum sysWrite)
    // has_side_effects prevents LLVM from optimizing away the syscall
    MLIRBuilder.line builder (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"=r,{rax},{rdi},{rsi},{rdx}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64"
        result sysNum fdVal bufPtr len)
    result

/// Emit syscall for Linux read
let emitSyscallRead (builder: MLIRBuilder) (ctx: SSAContext) (fd: string) (bufPtr: string) (len: string) =
    let sysRead = "0"  // Linux x86_64 read syscall number
    let fdVal = SSAContext.nextValue ctx
    let sysNum = SSAContext.nextValue ctx
    let result = SSAContext.nextValue ctx

    MLIRBuilder.line builder (sprintf "%s = arith.constant %s : i64" fdVal fd)
    MLIRBuilder.line builder (sprintf "%s = arith.constant %s : i64" sysNum sysRead)
    // has_side_effects prevents LLVM from optimizing away the syscall
    MLIRBuilder.line builder (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"=r,{rax},{rdi},{rsi},{rdx}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64"
        result sysNum fdVal bufPtr len)
    result

/// Emit a string constant (returns pointer and length)
let emitStringConstant (builder: MLIRBuilder) (ctx: SSAContext) (content: string) : string * string =
    let globalName = SSAContext.addStringLiteral ctx content
    let ptrVal = SSAContext.nextValue ctx
    let lenVal = SSAContext.nextValue ctx

    MLIRBuilder.line builder (sprintf "%s = llvm.mlir.addressof %s : !llvm.ptr" ptrVal globalName)
    MLIRBuilder.line builder (sprintf "%s = arith.constant %d : i64" lenVal content.Length)
    (ptrVal, lenVal)

/// Emit Console.write operation
let emitConsoleWrite (builder: MLIRBuilder) (ctx: SSAContext) (strContent: string) =
    let (ptr, len) = emitStringConstant builder ctx strContent
    emitSyscallWrite builder ctx "1" ptr len |> ignore  // fd=1 is stdout

/// Emit Console.writeLine operation
let emitConsoleWriteLine (builder: MLIRBuilder) (ctx: SSAContext) (strContent: string) =
    emitConsoleWrite builder ctx (strContent + "\n")

/// Check if a PSG node represents a Console operation
let isConsoleOperation (node: PSGNode) : (string * string option) option =
    match node.Symbol with
    | Some symbol ->
        let fullName = symbol.FullName
        if fullName = "Alloy.Console.write" || fullName.EndsWith(".Console.write") then
            Some ("write", None)
        elif fullName = "Alloy.Console.writeLine" || fullName.EndsWith(".Console.writeLine") then
            Some ("writeLine", None)
        elif fullName = "Alloy.Console.readLine" || fullName.EndsWith(".Console.readLine") then
            Some ("readLine", None)
        else None
    | None -> None

/// Check if a PSG node represents NativePtr.stackalloc
let isStackAllocOperation (node: PSGNode) : (string * int) option =
    // Look for TypeApp:byte or similar with stackalloc symbol
    match node.Symbol with
    | Some symbol ->
        let fullName = symbol.FullName
        if fullName.Contains("NativePtr.stackalloc") || fullName.Contains("stackalloc") then
            // Extract element type from SyntaxKind like "TypeApp:byte"
            let elemType =
                if node.SyntaxKind.StartsWith("TypeApp:") then
                    node.SyntaxKind.Substring(8)  // "byte", "int", etc.
                else "i8"  // Default to byte
            Some (elemType, 0)  // Size will be extracted from argument
        else None
    | None ->
        // Also check SyntaxKind for TypeApp nodes that might not have symbol yet
        if node.SyntaxKind.StartsWith("TypeApp:") then
            // Could be stackalloc - check children
            None
        else None

/// Get element size in bytes for a type name
let getElementSize (typeName: string) : int =
    match typeName.ToLowerInvariant() with
    | "byte" | "sbyte" | "int8" | "uint8" | "i8" -> 1
    | "int16" | "uint16" | "i16" -> 2
    | "int" | "int32" | "uint32" | "i32" -> 4
    | "int64" | "uint64" | "i64" | "nativeint" | "unativeint" -> 8
    | "float32" | "single" | "f32" -> 4
    | "float" | "float64" | "double" | "f64" -> 8
    | _ -> 1  // Default to 1 byte

/// Check if node is an application (function call)
let isApplicationNode (node: PSGNode) : bool =
    node.SyntaxKind.StartsWith("App") || node.SyntaxKind = "Application"

/// Check if node is a constant
let isConstNode (node: PSGNode) : bool =
    node.SyntaxKind.StartsWith("Const:")

/// Extract constant value from node
let getConstValue (node: PSGNode) : string option =
    if node.SyntaxKind.StartsWith("Const:String:") then
        // Extract string content from SyntaxKind like "Const:String:Hello"
        let parts = node.SyntaxKind.Split(':')
        if parts.Length >= 3 then
            Some (String.Join(":", parts.[2..]))
        else None
    elif node.SyntaxKind.StartsWith("Const:Int32:") then
        let parts = node.SyntaxKind.Split(':')
        if parts.Length >= 3 then Some parts.[2] else None
    elif node.SyntaxKind = "Const:Unit" then
        Some "()"
    else None

/// Walk children of a node
/// Note: Children are stored in reverse order (prepended during construction),
/// so we reverse them to get source order
let getChildren (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode list =
    match node.Children with
    | Parent childIds ->
        childIds
        |> List.rev  // Reverse to get source order
        |> List.choose (fun id -> Map.tryFind id.Value psg.Nodes)
    | _ -> []

/// Find the parent binding node that contains a function definition
let findParentBinding (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode option =
    // Walk up the parent chain to find a Binding node
    let rec walkUp current =
        match current.ParentId with
        | None -> None
        | Some parentId ->
            match Map.tryFind parentId.Value psg.Nodes with
            | Some parent ->
                if parent.SyntaxKind.StartsWith("Binding") || parent.SyntaxKind = "LetOrUse" then
                    Some parent
                else
                    walkUp parent
            | None -> None
    walkUp node

/// Find the main function from symbol table
let findMainFunction (psg: ProgramSemanticGraph) : PSGNode option =
    // First try to find "main" in symbol table
    match Map.tryFind "main" psg.SymbolTable with
    | Some mainSymbol ->
        // Find the node that has this symbol
        let mainNode =
            psg.Nodes
            |> Map.toList
            |> List.map snd
            |> List.tryFind (fun node ->
                match node.Symbol with
                | Some sym -> sym.FullName = mainSymbol.FullName
                | None -> false)
        // If main is a pattern node, find the parent binding
        match mainNode with
        | Some node when node.SyntaxKind.StartsWith("Pattern:") ->
            // Try to get the parent binding
            match findParentBinding psg node with
            | Some parentBinding -> Some parentBinding
            | None -> mainNode  // Fall back to the pattern node
        | other -> other
    | None ->
        // Look for any entry point binding
        psg.EntryPoints
        |> List.choose (fun epId -> Map.tryFind epId.Value psg.Nodes)
        |> List.tryHead

/// Find entry point functions in PSG
let findEntryPoints (psg: ProgramSemanticGraph) : PSGNode list =
    // First, look for the explicit main function
    match findMainFunction psg with
    | Some mainNode -> [mainNode]
    | None ->
        // Fall back to entry point list
        psg.EntryPoints
        |> List.choose (fun epId -> Map.tryFind epId.Value psg.Nodes)

/// Find all reachable function definitions
let findReachableFunctions (psg: ProgramSemanticGraph) : PSGNode list =
    psg.Nodes
    |> Map.toList
    |> List.map snd
    |> List.filter (fun node ->
        node.IsReachable &&
        (node.SyntaxKind = "Binding" ||
         node.SyntaxKind = "Binding:EntryPoint" ||
         node.SyntaxKind = "Binding:Main" ||
         node.SyntaxKind.StartsWith("LetBinding")))
    |> List.filter (fun node ->
        match node.Symbol with
        | Some sym ->
            match sym with
            | :? FSharpMemberOrFunctionOrValue as mfv -> mfv.IsFunction || mfv.IsMember
            | _ -> false
        | None -> false)

/// Emit function body by walking children
let rec emitExpression (builder: MLIRBuilder) (ctx: SSAContext) (psg: ProgramSemanticGraph) (node: PSGNode) : string option =
    match node.SyntaxKind with
    | sk when sk.StartsWith("Const:String:") ->
        let content = sk.Substring("Const:String:".Length)
        let (ptr, _) = emitStringConstant builder ctx content
        Some ptr

    | sk when sk.StartsWith("Const:Int32:") ->
        let value = sk.Substring("Const:Int32:".Length)
        let result = SSAContext.nextValue ctx
        MLIRBuilder.line builder (sprintf "%s = arith.constant %s : i32" result value)
        Some result

    | "Const:Unit" ->
        None  // Unit has no value

    | "Sequential" ->
        // Process children in order, return last result
        let children = getChildren psg node
        let mutable lastResult = None
        for child in children do
            lastResult <- emitExpression builder ctx psg child
        lastResult

    | sk when sk.StartsWith("App") || sk = "Application" ->
        // Function application - check for special operations first
        let children = getChildren psg node

        // Try to find what function is being called (TypeApp or LongIdent)
        let callTarget =
            children
            |> List.tryFind (fun c ->
                c.SyntaxKind.StartsWith("TypeApp:") ||
                c.SyntaxKind.StartsWith("LongIdent") ||
                c.SyntaxKind.StartsWith("Ident") ||
                c.SyntaxKind.StartsWith("App"))  // Nested app for curried calls

        // Check if this is NativePtr.stackalloc or Alloy.Memory.stackBuffer
        let isStackAlloc =
            match callTarget with
            | Some target ->
                match target.Symbol with
                | Some sym ->
                    let fullName = sym.FullName
                    fullName.Contains("NativePtr.stackalloc") ||
                    fullName.Contains("stackalloc") ||
                    fullName.Contains("stackBuffer") ||
                    sym.DisplayName = "stackBuffer"
                | None ->
                    target.SyntaxKind.StartsWith("TypeApp:") ||  // TypeApp with no symbol could be stackalloc
                    target.SyntaxKind.Contains("stackBuffer")
            | None -> false

        // Check if this is a Console operation
        let isConsole =
            match callTarget with
            | Some target ->
                match target.Symbol with
                | Some sym ->
                    let fullName = sym.FullName
                    fullName.StartsWith("Alloy.Console.") || fullName.Contains(".Console.")
                | None ->
                    target.SyntaxKind.Contains("Console")
            | None -> false

        // Check if this is an arithmetic operator
        // Check both Symbol.DisplayName and SyntaxKind for operator detection
        let arithmeticOp =
            match callTarget with
            | Some target ->
                // First try Symbol if available
                let fromSymbol =
                    match target.Symbol with
                    | Some sym ->
                        match sym.DisplayName with
                        | "op_Subtraction" -> Some "subi"
                        | "op_Addition" -> Some "addi"
                        | "op_Multiply" -> Some "muli"
                        | "op_Division" -> Some "divsi"
                        | "op_Modulus" -> Some "remsi"
                        | _ -> None
                    | None -> None

                // Fallback: check SyntaxKind for operator names
                match fromSymbol with
                | Some op -> Some op
                | None ->
                    if target.SyntaxKind.Contains("op_Subtraction") then Some "subi"
                    elif target.SyntaxKind.Contains("op_Addition") then Some "addi"
                    elif target.SyntaxKind.Contains("op_Multiply") then Some "muli"
                    elif target.SyntaxKind.Contains("op_Division") then Some "divsi"
                    elif target.SyntaxKind.Contains("op_Modulus") then Some "remsi"
                    else None
            | None -> None

        if isStackAlloc then
            // NativePtr.stackalloc<T> size -> llvm.alloca
            // Extract element type from TypeApp:byte
            let elemType =
                match callTarget with
                | Some target when target.SyntaxKind.StartsWith("TypeApp:") ->
                    target.SyntaxKind.Substring(8)  // "byte", "int", etc.
                | _ -> "byte"

            // Find size argument (should be a Const node)
            let sizeArg = children |> List.tryFind (fun c -> c.SyntaxKind.StartsWith("Const:Int"))
            let size =
                match sizeArg with
                | Some arg ->
                    match getConstValue arg with
                    | Some s -> int s
                    | None -> 64  // Default
                | None -> 64

            // Map F# type to LLVM type
            let llvmElemType =
                match elemType.ToLowerInvariant() with
                | "byte" | "sbyte" -> "i8"
                | "int16" | "uint16" -> "i16"
                | "int" | "int32" | "uint32" -> "i32"
                | "int64" | "uint64" -> "i64"
                | _ -> "i8"

            // Emit stack allocation
            // llvm.alloca requires SSA value for count, not inline literal
            let sizeVal = SSAContext.nextValue ctx
            MLIRBuilder.line builder (sprintf "%s = llvm.mlir.constant(%d : i64) : i64" sizeVal size)
            let ptr = SSAContext.nextValue ctx
            MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x %s : (i64) -> !llvm.ptr" ptr sizeVal llvmElemType)
            Some ptr

        elif isConsole then
            // Identify which Console operation
            let opName =
                match callTarget with
                | Some target ->
                    match target.Symbol with
                    | Some sym -> sym.DisplayName
                    | None ->
                        if target.SyntaxKind.Contains("writeLine") then "writeLine"
                        elif target.SyntaxKind.Contains("write") then "write"
                        elif target.SyntaxKind.Contains("readLine") then "readLine"
                        else "unknown"
                | None -> "unknown"

            // Recursively find string argument in children
            let rec findStringArg nodes depth =
                match nodes with
                | [] -> None
                | node :: rest ->
                    // Check for various string constant formats
                    // PSG format is: Const:String ("content", Regular, ...)
                    if node.SyntaxKind.StartsWith("Const:String") then
                        Some node
                    elif node.SyntaxKind.StartsWith("Const:\"") then
                        Some node
                    else
                        let nodeChildren = getChildren psg node
                        match findStringArg nodeChildren (depth + 1) with
                        | Some found -> Some found
                        | None -> findStringArg rest depth

            let strArg = findStringArg children 0

            // Extract string content from PSG syntax kind format
            // Format: Const:String ("content", Regular, range)
            let extractStringContent (syntaxKind: string) =
                // Try to extract the quoted string from the format
                let startQuote = syntaxKind.IndexOf("(\"")
                if startQuote >= 0 then
                    let contentStart = startQuote + 2
                    let endQuote = syntaxKind.IndexOf("\",", contentStart)
                    if endQuote > contentStart then
                        Some (syntaxKind.Substring(contentStart, endQuote - contentStart))
                    else None
                else None

            match opName with
            // === FidelityHelloWorld Console patterns ===
            | "WriteLine" ->
                // FidelityHelloWorld: Console.WriteLine message
                match strArg with
                | Some strNode ->
                    match extractStringContent strNode.SyntaxKind with
                    | Some content ->
                        emitConsoleWriteLine builder ctx content
                    | None ->
                        MLIRBuilder.line builder (sprintf "// Console.WriteLine - could not extract: %s" strNode.SyntaxKind)
                | None ->
                    MLIRBuilder.line builder "// Console.WriteLine with non-literal arg"
                None
            | "Write" | "Prompt" ->
                // FidelityHelloWorld: Console.Write / Prompt message
                match strArg with
                | Some strNode ->
                    match extractStringContent strNode.SyntaxKind with
                    | Some content ->
                        emitConsoleWrite builder ctx content
                    | None ->
                        MLIRBuilder.line builder (sprintf "// Console.Write - could not extract: %s" strNode.SyntaxKind)
                | None ->
                    MLIRBuilder.line builder "// Console.Write with non-literal arg"
                None
            | "writeBytes" ->
                // FidelityHelloWorld primitive: writeBytes fd buffer count
                // This is the actual syscall - emit inline assembly
                MLIRBuilder.line builder "// writeBytes: Firefly primitive syscall"
                // For now, emit placeholder - full implementation needs fd, buffer, count args
                None
            | "readBytes" ->
                // FidelityHelloWorld primitive: readBytes fd buffer maxCount
                MLIRBuilder.line builder "// readBytes: Firefly primitive syscall"
                None
            | "readInto" ->
                // FidelityHelloWorld: readInto buffer (SRTP-based)
                // Finds buffer arg and emits read syscall
                let bufferArg = children |> List.tryFind (fun c ->
                    c.SyntaxKind.StartsWith("Ident") &&
                    not (c.SyntaxKind.Contains("readInto")))
                match bufferArg with
                | Some buf ->
                    match buf.Symbol with
                    | Some sym ->
                        match SSAContext.lookupLocal ctx sym.DisplayName with
                        | Some bufPtr ->
                            // Emit read syscall for stdin (fd=0)
                            let size = SSAContext.nextValue ctx
                            MLIRBuilder.line builder (sprintf "%s = arith.constant 64 : i64" size)
                            let fd = SSAContext.nextValue ctx
                            MLIRBuilder.line builder (sprintf "%s = arith.constant 0 : i64" fd)
                            let sysRead = SSAContext.nextValue ctx
                            MLIRBuilder.line builder (sprintf "%s = arith.constant 0 : i64" sysRead)
                            let result = SSAContext.nextValue ctx
                            MLIRBuilder.line builder (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"=r,{rax},{rdi},{rsi},{rdx}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64" result sysRead fd bufPtr size)
                            let truncResult = SSAContext.nextValue ctx
                            MLIRBuilder.line builder (sprintf "%s = arith.trunci %s : i64 to i32" truncResult result)
                            Some truncResult
                        | None ->
                            MLIRBuilder.line builder (sprintf "// readInto: buffer '%s' not found" sym.DisplayName)
                            None
                    | None ->
                        MLIRBuilder.line builder "// readInto: buffer has no symbol"
                        None
                | None ->
                    MLIRBuilder.line builder "// readInto: no buffer argument found"
                    None
            // === Legacy Alloy Console patterns ===
            | "writeLine" ->
                match strArg with
                | Some strNode ->
                    match extractStringContent strNode.SyntaxKind with
                    | Some content ->
                        emitConsoleWriteLine builder ctx content
                    | None ->
                        MLIRBuilder.line builder (sprintf "// Console.writeLine - could not extract: %s" strNode.SyntaxKind)
                | None ->
                    MLIRBuilder.line builder "// Console.writeLine with non-literal arg"
                None
            | "write" ->
                match strArg with
                | Some strNode ->
                    match extractStringContent strNode.SyntaxKind with
                    | Some content ->
                        emitConsoleWrite builder ctx content
                    | None ->
                        MLIRBuilder.line builder (sprintf "// Console.write - could not extract: %s" strNode.SyntaxKind)
                | None ->
                    MLIRBuilder.line builder "// Console.write with non-literal arg"
                None
            | "writeBuffer" ->
                // Console.writeBuffer buffer length -> write syscall to stdout
                // This is handled specially because F# curried calls create nested Apps.
                // When we detect writeBuffer at the inner App level, we only have the buffer arg.
                // The length arg is at the outer App level.
                // We return a "partial" indicator and let the outer App complete the call.

                // Debug: show all children
                let childDescs = children |> List.map (fun c ->
                    let symName = match c.Symbol with | Some s -> s.DisplayName | None -> "?"
                    sprintf "%s:%s" c.SyntaxKind symName)
                MLIRBuilder.line builder (sprintf "// writeBuffer children: %s" (String.concat ", " childDescs))

                // Debug: show parent chain
                let rec showParents n depth =
                    if depth > 5 then ()
                    else
                        match n.ParentId with
                        | Some pid ->
                            match Map.tryFind pid.Value psg.Nodes with
                            | Some parent ->
                                let parentChildren = getChildren psg parent
                                let pChildDescs = parentChildren |> List.map (fun c -> c.SyntaxKind) |> String.concat ", "
                                MLIRBuilder.line builder (sprintf "// Parent[%d]: %s children=[%s]" depth parent.SyntaxKind pChildDescs)
                                showParents parent (depth + 1)
                            | None -> ()
                        | None -> ()
                showParents node 0

                // Find buffer argument - must be an Ident that is NOT the writeBuffer function itself
                let bufferArg = children |> List.tryFind (fun c ->
                    // Must be an Ident (not LongIdent, which would be Console.writeBuffer)
                    c.SyntaxKind.StartsWith("Ident") &&
                    not (c.SyntaxKind.Contains("writeBuffer")) &&
                    match c.Symbol with
                    | Some sym ->
                        // Exclude the writeBuffer function reference
                        not (sym.DisplayName = "writeBuffer") &&
                        not (sym.FullName.Contains("Console"))
                    | None -> true)

                // Filter args to exclude the function reference
                let args = children |> List.filter (fun c ->
                    match c.Symbol with
                    | Some sym ->
                        not (sym.FullName.Contains("Console.writeBuffer") ||
                             sym.FullName.Contains("Console"))
                    | None -> not (c.SyntaxKind.Contains("writeBuffer") || c.SyntaxKind.Contains("Console")))

                MLIRBuilder.line builder (sprintf "// writeBuffer args after filter: %s"
                    (args |> List.map (fun c -> c.SyntaxKind) |> String.concat ", "))

                match bufferArg with
                | Some arg ->
                    match arg.Symbol with
                    | Some sym ->
                        MLIRBuilder.line builder (sprintf "// writeBuffer buffer arg symbol: %s" sym.DisplayName)
                        match SSAContext.lookupLocal ctx sym.DisplayName with
                        | Some ssaName ->
                            // Store the buffer pointer for the outer App to use
                            SSAContext.registerLocal ctx "__writeBuffer_ptr" ssaName
                            // Return a marker that this is a partial writeBuffer
                            Some "__writeBuffer_partial"
                        | None ->
                            MLIRBuilder.line builder (sprintf "// Warning: buffer '%s' not found in locals" sym.DisplayName)
                            None
                    | None ->
                        MLIRBuilder.line builder "// Warning: buffer arg has no symbol"
                        None
                | None ->
                    MLIRBuilder.line builder "// Warning: no buffer argument found"
                    None

            | "readLine" ->
                // Console.readLine buffer size -> read syscall from stdin
                // Arguments: buffer (pointer), size (int)

                // Filter out the call target from children to get just the arguments
                let args = children |> List.filter (fun c ->
                    match c.Symbol with
                    | Some sym ->
                        // Skip the function itself and module references
                        let fn = sym.FullName
                        not (fn.Contains("Console.readLine") || fn.Contains("Console"))
                    | None ->
                        // Keep nodes without symbols that might be arguments
                        not (c.SyntaxKind.Contains("readLine") || c.SyntaxKind.Contains("Console")))

                // Find buffer argument (should be an Ident referencing a local variable)
                let bufferArg = args |> List.tryFind (fun c ->
                    c.SyntaxKind.StartsWith("Ident") ||
                    c.SyntaxKind.StartsWith("LongIdent"))

                // Find size argument (should be a Const node)
                let sizeArg = args |> List.tryFind (fun c -> c.SyntaxKind.StartsWith("Const:Int"))

                // Get buffer pointer from local variable lookup
                let bufPtr =
                    match bufferArg with
                    | Some arg ->
                        match arg.Symbol with
                        | Some sym ->
                            match SSAContext.lookupLocal ctx sym.DisplayName with
                            | Some ssaName -> ssaName
                            | None ->
                                // Buffer not found - this is an error, but continue with dummy
                                MLIRBuilder.line builder (sprintf "// Warning: buffer '%s' not found in locals" sym.DisplayName)
                                let ptr = SSAContext.nextValue ctx
                                MLIRBuilder.line builder (sprintf "%s = llvm.alloca i8 x 64 : (i64) -> !llvm.ptr" ptr)
                                ptr
                        | None ->
                            let ptr = SSAContext.nextValue ctx
                            MLIRBuilder.line builder "// Warning: buffer argument has no symbol"
                            MLIRBuilder.line builder (sprintf "%s = llvm.alloca i8 x 64 : (i64) -> !llvm.ptr" ptr)
                            ptr
                    | None ->
                        let ptr = SSAContext.nextValue ctx
                        MLIRBuilder.line builder "// Warning: no buffer argument found"
                        MLIRBuilder.line builder (sprintf "%s = llvm.alloca i8 x 64 : (i64) -> !llvm.ptr" ptr)
                        ptr

                // Get size value
                let size =
                    match sizeArg with
                    | Some arg ->
                        match getConstValue arg with
                        | Some s -> int s
                        | None -> 64
                    | None -> 64

                // Emit size constant
                let bufSize = SSAContext.nextValue ctx
                MLIRBuilder.line builder (sprintf "%s = arith.constant %d : i64" bufSize size)

                // Emit read syscall: read(fd=0, buf, count)
                let bytesRead = emitSyscallRead builder ctx "0" bufPtr bufSize

                // Return bytes read (truncated to i32)
                let result = SSAContext.nextValue ctx
                MLIRBuilder.line builder (sprintf "%s = arith.trunci %s : i64 to i32" result bytesRead)
                Some result
            | _ ->
                MLIRBuilder.line builder (sprintf "// Unknown Console operation: %s" opName)
                None

        elif arithmeticOp.IsSome then
            // Arithmetic operation: op_Subtraction, op_Addition, etc.
            let op = arithmeticOp.Value

            // Get operands - filter out the operator itself (check both Symbol AND SyntaxKind)
            let operands = children |> List.filter (fun c ->
                let isOperatorBySymbol =
                    match c.Symbol with
                    | Some sym -> sym.DisplayName.StartsWith("op_")
                    | None -> false
                let isOperatorBySyntax = c.SyntaxKind.Contains("op_")
                // Filter out if EITHER indicates it's an operator
                not (isOperatorBySymbol || isOperatorBySyntax))

            // Debug
            let operandDescs = operands |> List.map (fun c -> c.SyntaxKind) |> String.concat ", "
            MLIRBuilder.line builder (sprintf "// Arithmetic operands: [%s]" operandDescs)

            // Emit both operands
            let operandValues = operands |> List.choose (fun operand ->
                emitExpression builder ctx psg operand)

            MLIRBuilder.line builder (sprintf "// Arithmetic operand values: %d" operandValues.Length)

            match operandValues with
            | [lhs; rhs] ->
                let result = SSAContext.nextValue ctx
                MLIRBuilder.line builder (sprintf "%s = arith.%s %s, %s : i32" result op lhs rhs)
                Some result
            | [single] ->
                // Might be a unary operation or partially applied - return it for outer App
                // This handles curried arithmetic like (bytesRead - 1) where inner App has just bytesRead
                Some single
            | _ ->
                MLIRBuilder.line builder (sprintf "// Arithmetic op %s with unexpected operand count: %d" op operandValues.Length)
                None

        else
            // Check if this is the outer App completing a partial writeBuffer call
            // F# curried calls: Console.writeBuffer buffer length becomes App(App(writeBuffer, buffer), length)
            // The inner App returns "__writeBuffer_partial", and we complete the syscall here with the length arg

            // First, try emitting the first child (which may be the inner App with writeBuffer)
            let firstChildResult =
                match children |> List.tryHead with
                | Some firstChild -> emitExpression builder ctx psg firstChild
                | None -> None

            // Debug: show what we have
            let childDescs = children |> List.map (fun c -> c.SyntaxKind) |> String.concat ", "
            MLIRBuilder.line builder (sprintf "// Outer App children (%d): [%s]" children.Length childDescs)
            MLIRBuilder.line builder (sprintf "// firstChildResult = %A" firstChildResult)

            // Check if the first child returned the partial writeBuffer marker
            if firstChildResult = Some "__writeBuffer_partial" then
                // This is the outer App - complete the writeBuffer syscall
                // Get the stored buffer pointer
                match SSAContext.lookupLocal ctx "__writeBuffer_ptr" with
                | Some bufPtr ->
                    // Get the second child (the length expression)
                    let lengthArg = children |> List.tryItem 1
                    match lengthArg with
                    | Some lengthNode ->
                        // Emit the length expression (could be arithmetic like bytesRead - 1)
                        let lengthResult = emitExpression builder ctx psg lengthNode
                        match lengthResult with
                        | Some lengthVal ->
                            // Extend length to i64 for syscall
                            let len64 = SSAContext.nextValue ctx
                            MLIRBuilder.line builder (sprintf "%s = arith.extsi %s : i32 to i64" len64 lengthVal)

                            // Emit write syscall: write(fd=1, buf, len)
                            let fd = SSAContext.nextValue ctx
                            MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i64" fd)
                            let sysWrite = SSAContext.nextValue ctx
                            MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i64" sysWrite)
                            let result = SSAContext.nextValue ctx
                            MLIRBuilder.line builder (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"=r,{rax},{rdi},{rsi},{rdx}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64" result sysWrite fd bufPtr len64)
                            Some result
                        | None ->
                            MLIRBuilder.line builder "// writeBuffer: failed to emit length expression"
                            None
                    | None ->
                        MLIRBuilder.line builder "// writeBuffer: no length argument found"
                        None
                | None ->
                    MLIRBuilder.line builder "// writeBuffer: buffer pointer not found in context"
                    None
            else
                // Not a partial writeBuffer - continue with generic App handling
                // Generic function call - but skip module/type references
                let isModuleOrType =
                    match callTarget with
                    | Some target ->
                        match target.Symbol with
                        | Some sym ->
                            match sym with
                            | :? FSharpEntity as entity -> entity.IsFSharpModule
                            | _ -> false
                        | None -> false
                    | None -> false

                if isModuleOrType then
                    // Skip module references - just emit children
                    let children = getChildren psg node
                    let mutable lastResult = None
                    for child in children do
                        if not (child.SyntaxKind.StartsWith("LongIdent") || child.SyntaxKind.StartsWith("Ident")) then
                            lastResult <- emitExpression builder ctx psg child
                    lastResult
                else
                    match node.Symbol with
                    | Some sym ->
                        match sym with
                        | :? FSharpEntity as entity when entity.IsFSharpModule ->
                            // Skip module references
                            None
                        | :? FSharpMemberOrFunctionOrValue as mfv ->
                            let funcName = sym.DisplayName.Replace(".", "_")
                            let result = SSAContext.nextValue ctx
                            MLIRBuilder.line builder (sprintf "%s = func.call @%s() : () -> i32" result funcName)
                            Some result
                        | _ ->
                            MLIRBuilder.line builder (sprintf "// Skipping non-function symbol: %s" sym.DisplayName)
                            None
                    | None ->
                        // Try to get function name from call target
                        match callTarget with
                        | Some target ->
                            match target.Symbol with
                            | Some sym ->
                                match sym with
                                | :? FSharpMemberOrFunctionOrValue ->
                                    let funcName = sym.DisplayName.Replace(".", "_")
                                    let result = SSAContext.nextValue ctx
                                    MLIRBuilder.line builder (sprintf "%s = func.call @%s() : () -> i32" result funcName)
                                    Some result
                                | _ ->
                                    MLIRBuilder.line builder (sprintf "// Skipping non-function: %s" sym.DisplayName)
                                    None
                            | None ->
                                MLIRBuilder.line builder (sprintf "// Unresolved function call: %s" target.SyntaxKind)
                                None
                        | None ->
                            MLIRBuilder.line builder "// Unresolved function call"
                            None

    | sk when sk.StartsWith("Ident") ->
        // Variable reference - SyntaxKind may be "Ident" or "Ident:varname"
        let varName =
            match node.Symbol with
            | Some sym -> Some sym.DisplayName
            | None ->
                // Extract from SyntaxKind like "Ident:bytesRead"
                if sk.StartsWith("Ident:") then Some (sk.Substring(6))
                else None

        match varName with
        | Some name ->
            match SSAContext.lookupLocal ctx name with
            | Some ssaName -> Some ssaName
            | None ->
                // Might be a function reference - emit as function call
                let result = SSAContext.nextValue ctx
                MLIRBuilder.line builder (sprintf "%s = func.call @%s() : () -> i32" result name)
                Some result
        | None ->
            MLIRBuilder.line builder (sprintf "// Ident with no name: %s" sk)
            None

    | sk when sk = "LetOrUse" || sk = "Binding" || sk.StartsWith("Binding:") ->
        // Let binding - extract variable name, emit expression, register SSA name
        let children = getChildren psg node

        // Find pattern node (variable name) and expression
        let patternNode = children |> List.tryFind (fun c -> c.SyntaxKind.StartsWith("Pattern:"))
        let exprNode = children |> List.tryFind (fun c ->
            not (c.SyntaxKind.StartsWith("Pattern:")) &&
            not (c.SyntaxKind.StartsWith("Attribute")))

        // Get variable name from pattern
        let varName =
            match patternNode with
            | Some pat ->
                match pat.Symbol with
                | Some sym -> Some sym.DisplayName
                | None ->
                    // Try to extract from SyntaxKind like "Pattern:Named:buffer"
                    if pat.SyntaxKind.StartsWith("Pattern:Named:") then
                        Some (pat.SyntaxKind.Substring(14))
                    else None
            | None -> None

        // Emit the expression
        match exprNode with
        | Some expr ->
            let result = emitExpression builder ctx psg expr
            // Register the SSA name if we have a variable name
            match varName, result with
            | Some name, Some ssaName ->
                MLIRBuilder.line builder (sprintf "// Registering local: %s = %s" name ssaName)
                SSAContext.registerLocal ctx name ssaName
            | Some name, None ->
                MLIRBuilder.line builder (sprintf "// Warning: no SSA result for local: %s" name)
            | None, _ ->
                MLIRBuilder.line builder (sprintf "// Warning: no var name for binding")
            result
        | None ->
            // Fall back to emitting all children
            let mutable lastResult = None
            for child in children do
                lastResult <- emitExpression builder ctx psg child
            lastResult

    | "LongIdentSet" | "Set" ->
        // Mutable variable assignment
        let children = getChildren psg node
        if children.Length >= 1 then
            emitExpression builder ctx psg (List.last children)
        else None

    | _ ->
        // Unknown node - try to emit children
        let children = getChildren psg node
        if children.Length > 0 then
            let mutable lastResult = None
            for child in children do
                lastResult <- emitExpression builder ctx psg child
            lastResult
        else
            MLIRBuilder.line builder (sprintf "// Unhandled node: %s" node.SyntaxKind)
            None

/// Emit a function definition
let emitFunction (builder: MLIRBuilder) (ctx: SSAContext) (psg: ProgramSemanticGraph) (node: PSGNode) (isEntryPoint: bool) =
    // Handle entry point with attribute symbol by using default signature
    let emitWithParams funcName funcParams retType =
        let paramStr =
            if funcParams = [] then ""
            else
                funcParams
                |> List.mapi (fun i (name, typ) -> sprintf "%%arg%d: %s" i typ)
                |> String.concat ", "

        // For entry points, always return i32
        let actualRetType = if isEntryPoint then "i32" else retType

        MLIRBuilder.line builder ""
        MLIRBuilder.line builder (sprintf "func.func @%s(%s) -> %s {" funcName paramStr actualRetType)
        MLIRBuilder.push builder

        // Bind parameters to SSA names
        funcParams |> List.iteri (fun i (name: string, _) ->
            ctx.Locals <- Map.add name (sprintf "%%arg%d" i) ctx.Locals)

        // Emit function body
        let children = getChildren psg node
        let mutable lastResult = None
        for child in children do
            // Skip pattern nodes (function name pattern)
            if not (child.SyntaxKind.StartsWith("Pattern:")) then
                lastResult <- emitExpression builder ctx psg child

        // Emit return
        match lastResult with
        | Some result when actualRetType = "i32" ->
            MLIRBuilder.line builder (sprintf "return %s : i32" result)
        | _ ->
            if actualRetType = "()" then
                MLIRBuilder.line builder "return"
            else
                let zero = SSAContext.nextValue ctx
                MLIRBuilder.line builder (sprintf "%s = arith.constant 0 : i32" zero)
                MLIRBuilder.line builder (sprintf "return %s : i32" zero)

        MLIRBuilder.pop builder
        MLIRBuilder.line builder "}"

    match node.Symbol with
    | Some sym ->
        match sym with
        | :? FSharpMemberOrFunctionOrValue as mfv ->
            let funcName =
                if isEntryPoint then "main"
                else mfv.DisplayName.Replace(".", "_")

            let funcParams = getFunctionParamTypes mfv
            let retType = getFunctionReturnType mfv
            emitWithParams funcName funcParams retType

        | :? FSharpEntity as entity when entity.IsAttributeType ->
            // This is an entry point with attribute as symbol
            // Use default main signature: main(argc: i32) -> i32
            if isEntryPoint then
                emitWithParams "main" [] "i32"
            else
                ()  // Skip attribute-only nodes

        | _ ->
            // Unknown symbol type - emit with default signature if entry point
            if isEntryPoint then
                emitWithParams "main" [] "i32"

    | None ->
        // No symbol - emit with default signature if entry point
        if isEntryPoint then
            emitWithParams "main" [] "i32"

/// Emit string constant globals
let emitStringGlobals (builder: MLIRBuilder) (ctx: SSAContext) =
    for (content, name) in List.rev ctx.StringLiterals do
        let escaped =
            content
            |> String.collect (fun c ->
                match c with
                | '\n' -> "\\0A"
                | '\r' -> "\\0D"
                | '\t' -> "\\09"
                | '"' -> "\\22"
                | '\\' -> "\\\\"
                | c when int c < 32 || int c > 126 -> sprintf "\\%02X" (int c)
                | c -> string c)
        MLIRBuilder.line builder (sprintf "llvm.mlir.global private constant %s(\"%s\\00\") : !llvm.array<%d x i8>"
            name escaped (content.Length + 1))

/// Emit _start wrapper for freestanding binaries
/// This calls main and then invokes the exit syscall with the return value
/// We use func dialect so the --convert-to-llvm pass converts everything together
let emitStartWrapper (builder: MLIRBuilder) =
    MLIRBuilder.line builder ""
    MLIRBuilder.line builder "// Entry point wrapper for freestanding binary"
    MLIRBuilder.line builder "func.func @_start() attributes {llvm.emit_c_interface} {"
    MLIRBuilder.push builder
    // Call main with argc=0
    MLIRBuilder.line builder "%argc = arith.constant 0 : i32"
    MLIRBuilder.line builder "%retval = func.call @main(%argc) : (i32) -> i32"
    // Extend return value to i64 for syscall
    MLIRBuilder.line builder "%retval64 = arith.extsi %retval : i32 to i64"
    // exit syscall (60 on x86_64 Linux) - {rax} is syscall number, {rdi} is exit code
    // Mark as has_side_effects to prevent optimization
    MLIRBuilder.line builder "%sys_exit = arith.constant 60 : i64"
    MLIRBuilder.line builder "%unused = llvm.inline_asm has_side_effects \"syscall\", \"=r,{rax},{rdi}\" %sys_exit, %retval64 : (i64, i64) -> i64"
    // Use cf.br to self as infinite loop after exit - exit should never return
    // Actually just return - the exit syscall will terminate the process
    MLIRBuilder.line builder "return"
    MLIRBuilder.pop builder
    MLIRBuilder.line builder "}"

/// Main entry point: generate MLIR from PSG
let generateMLIR (psg: ProgramSemanticGraph) (projectName: string) (targetTriple: string) (outputKind: OutputKind) : string =
    let builder = MLIRBuilder.create()
    let ctx = SSAContext.create()

    // Header comments - MLIR uses // for comments
    MLIRBuilder.lineNoIndent builder (sprintf "// Firefly-generated MLIR for %s" projectName)
    MLIRBuilder.lineNoIndent builder (sprintf "// Target: %s" targetTriple)
    MLIRBuilder.lineNoIndent builder (sprintf "// Output: %s" (OutputKind.toString outputKind))
    MLIRBuilder.lineNoIndent builder (sprintf "// PSG: %d nodes, %d edges, %d entry points"
        psg.Nodes.Count psg.Edges.Length psg.EntryPoints.Length)
    MLIRBuilder.lineNoIndent builder ""

    // Module start
    MLIRBuilder.line builder "module {"
    MLIRBuilder.push builder

    // Find entry points
    let entryPoints = findEntryPoints psg
    let entryPointIds = psg.EntryPoints |> List.map (fun id -> id.Value) |> Set.ofList

    // Emit all reachable functions
    let functions = findReachableFunctions psg

    // Emit entry points first
    for ep in entryPoints do
        emitFunction builder ctx psg ep true

    // Emit other functions (excluding entry points to avoid duplicates)
    for func in functions do
        if not (Set.contains func.Id.Value entryPointIds) then
            emitFunction builder ctx psg func false

    // If no entry points, generate default main
    if entryPoints.IsEmpty && functions.IsEmpty then
        MLIRBuilder.line builder ""
        MLIRBuilder.line builder "// No entry point found - generating default"
        MLIRBuilder.line builder "func.func @main() -> i32 {"
        MLIRBuilder.push builder
        MLIRBuilder.line builder "%0 = arith.constant 0 : i32"
        MLIRBuilder.line builder "return %0 : i32"
        MLIRBuilder.pop builder
        MLIRBuilder.line builder "}"

    // For freestanding binaries, emit _start wrapper
    match outputKind with
    | Freestanding | Embedded ->
        emitStartWrapper builder
    | Console | Library ->
        ()  // No wrapper needed - libc provides _start (console) or no entry (library)

    // Emit string globals at the end
    if not ctx.StringLiterals.IsEmpty then
        MLIRBuilder.line builder ""
        MLIRBuilder.line builder "// String constants"
        emitStringGlobals builder ctx

    MLIRBuilder.pop builder
    MLIRBuilder.line builder "}"

    MLIRBuilder.toString builder
