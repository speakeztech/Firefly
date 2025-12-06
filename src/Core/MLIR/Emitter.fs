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
            | Some name -> sprintf "!opaque<%s>" (name.Replace(".", "_"))
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
    MLIRBuilder.line builder (sprintf "%s = llvm.inline_asm \"syscall\", \"=r,{rax},{rdi},{rsi},{rdx}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64"
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
    MLIRBuilder.line builder (sprintf "%s = llvm.inline_asm \"syscall\", \"=r,{rax},{rdi},{rsi},{rdx}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64"
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
let getChildren (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode list =
    match node.Children with
    | Parent childIds ->
        childIds
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
        // Function application - look for Console operations in the call target
        let children = getChildren psg node

        // Try to find what function is being called
        let callTarget =
            children
            |> List.tryFind (fun c ->
                c.SyntaxKind.StartsWith("LongIdent") ||
                c.SyntaxKind.StartsWith("Ident") ||
                c.SyntaxKind.StartsWith("App"))  // Nested app for curried calls

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

        if isConsole then
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
            | "writeLine" ->
                match strArg with
                | Some strNode ->
                    match extractStringContent strNode.SyntaxKind with
                    | Some content ->
                        emitConsoleWriteLine builder ctx content
                    | None ->
                        MLIRBuilder.line builder (sprintf "; Console.writeLine - could not extract: %s" strNode.SyntaxKind)
                | None ->
                    MLIRBuilder.line builder "; Console.writeLine with non-literal arg"
                None
            | "write" ->
                match strArg with
                | Some strNode ->
                    match extractStringContent strNode.SyntaxKind with
                    | Some content ->
                        emitConsoleWrite builder ctx content
                    | None ->
                        MLIRBuilder.line builder (sprintf "; Console.write - could not extract: %s" strNode.SyntaxKind)
                | None ->
                    MLIRBuilder.line builder "; Console.write with non-literal arg"
                None
            | "readLine" ->
                let result = SSAContext.nextValue ctx
                MLIRBuilder.line builder "; TODO: Console.readLine needs buffer handling"
                MLIRBuilder.line builder (sprintf "%s = arith.constant 0 : i32" result)
                Some result
            | _ ->
                MLIRBuilder.line builder (sprintf "; Unknown Console operation: %s" opName)
                None
        else
            // Generic function call
            match node.Symbol with
            | Some sym ->
                let funcName = sym.DisplayName.Replace(".", "_")
                let result = SSAContext.nextValue ctx
                MLIRBuilder.line builder (sprintf "%s = func.call @%s() : () -> i32" result funcName)
                Some result
            | None ->
                // Try to get function name from call target
                match callTarget with
                | Some target ->
                    match target.Symbol with
                    | Some sym ->
                        let funcName = sym.DisplayName.Replace(".", "_")
                        let result = SSAContext.nextValue ctx
                        MLIRBuilder.line builder (sprintf "%s = func.call @%s() : () -> i32" result funcName)
                        Some result
                    | None ->
                        MLIRBuilder.line builder (sprintf "; Unresolved function call: %s" target.SyntaxKind)
                        None
                | None ->
                    MLIRBuilder.line builder "; Unresolved function call"
                    None

    | "Ident" ->
        // Variable reference
        match node.Symbol with
        | Some sym ->
            match SSAContext.lookupLocal ctx sym.DisplayName with
            | Some ssaName -> Some ssaName
            | None ->
                // Might be a function reference
                let result = SSAContext.nextValue ctx
                MLIRBuilder.line builder (sprintf "; Reference to: %s" sym.FullName)
                Some result
        | None -> None

    | "LetOrUse" | "Binding" ->
        // Let binding - emit the bound expression
        let children = getChildren psg node
        match children with
        | [] -> None
        | [single] -> emitExpression builder ctx psg single
        | _ ->
            // Multiple children - pattern + expression
            // For now, just emit the last (the bound expression)
            emitExpression builder ctx psg (List.last children)

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
            MLIRBuilder.line builder (sprintf "; Unhandled node: %s" node.SyntaxKind)
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

/// Main entry point: generate MLIR from PSG
let generateMLIR (psg: ProgramSemanticGraph) (projectName: string) (targetTriple: string) : string =
    let builder = MLIRBuilder.create()
    let ctx = SSAContext.create()

    // Header comments
    MLIRBuilder.lineNoIndent builder (sprintf "; Firefly-generated MLIR for %s" projectName)
    MLIRBuilder.lineNoIndent builder (sprintf "; Target: %s" targetTriple)
    MLIRBuilder.lineNoIndent builder (sprintf "; PSG: %d nodes, %d edges, %d entry points"
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
        MLIRBuilder.line builder "; No entry point found - generating default"
        MLIRBuilder.line builder "func.func @main() -> i32 {"
        MLIRBuilder.push builder
        MLIRBuilder.line builder "%0 = arith.constant 0 : i32"
        MLIRBuilder.line builder "return %0 : i32"
        MLIRBuilder.pop builder
        MLIRBuilder.line builder "}"

    // Emit string globals at the end
    if not ctx.StringLiterals.IsEmpty then
        MLIRBuilder.line builder ""
        MLIRBuilder.line builder "; String constants"
        emitStringGlobals builder ctx

    MLIRBuilder.pop builder
    MLIRBuilder.line builder "}"

    MLIRBuilder.toString builder
