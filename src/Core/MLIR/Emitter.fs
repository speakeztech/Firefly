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
    mutable LocalTypes: Map<string, string>  // F# name -> MLIR type (i32, i64, etc.)
    mutable SSATypes: Map<string, string>  // SSA name -> MLIR type (for FIDELITY tracking)
    mutable StringLiterals: (string * string) list  // content -> global name
}

module SSAContext =
    let create() = { Counter = 0; Locals = Map.empty; LocalTypes = Map.empty; SSATypes = Map.empty; StringLiterals = [] }

    let nextValue ctx =
        let name = sprintf "%%v%d" ctx.Counter
        ctx.Counter <- ctx.Counter + 1
        name

    /// Create next SSA value and track its type - FIDELITY
    let nextValueWithType ctx mlirType =
        let name = sprintf "%%v%d" ctx.Counter
        ctx.Counter <- ctx.Counter + 1
        ctx.SSATypes <- Map.add name mlirType ctx.SSATypes
        name

    let bindLocal ctx fsharpName =
        let ssaName = nextValue ctx
        ctx.Locals <- Map.add fsharpName ssaName ctx.Locals
        ssaName

    let lookupLocal ctx name =
        Map.tryFind name ctx.Locals

    let lookupLocalType ctx name =
        Map.tryFind name ctx.LocalTypes

    /// Look up the type of an SSA value - FIDELITY
    let lookupSSAType ctx ssaName =
        Map.tryFind ssaName ctx.SSATypes

    let registerLocal ctx fsharpName ssaName =
        ctx.Locals <- Map.add fsharpName ssaName ctx.Locals

    let registerLocalWithType ctx fsharpName ssaName mlirType =
        ctx.Locals <- Map.add fsharpName ssaName ctx.Locals
        ctx.LocalTypes <- Map.add fsharpName mlirType ctx.LocalTypes
        ctx.SSATypes <- Map.add ssaName mlirType ctx.SSATypes  // Also track SSA->type

    /// Register just the SSA value type - FIDELITY
    let registerSSAType ctx ssaName mlirType =
        ctx.SSATypes <- Map.add ssaName mlirType ctx.SSATypes

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

/// Check if a PSG node represents a Time operation
let isTimeOperation (node: PSGNode) : string option =
    match node.Symbol with
    | Some symbol ->
        let fullName = symbol.FullName
        // Check for Time module patterns
        if fullName.Contains("Time.currentTicks") || fullName.Contains("TimeApi.currentTicks") ||
           fullName.EndsWith(".currentTicks") then
            Some "currentTicks"
        elif fullName.Contains("Time.highResolutionTicks") || fullName.Contains("TimeApi.highResolutionTicks") ||
             fullName.EndsWith(".highResolutionTicks") then
            Some "highResolutionTicks"
        elif fullName.Contains("Time.tickFrequency") || fullName.Contains("TimeApi.tickFrequency") ||
             fullName.EndsWith(".tickFrequency") then
            Some "tickFrequency"
        elif fullName.Contains("Time.sleep") || fullName.Contains("TimeApi.sleep") ||
             fullName.EndsWith(".sleep") then
            Some "sleep"
        elif fullName.Contains("Time.currentUnixTimestamp") || fullName.Contains("TimeApi.currentUnixTimestamp") ||
             fullName.EndsWith(".currentUnixTimestamp") then
            Some "currentUnixTimestamp"
        else None
    | None -> None

/// Check if a PSG node represents a Text.Format operation
let isTextFormatOperation (node: PSGNode) : string option =
    match node.Symbol with
    | Some symbol ->
        let fullName = symbol.FullName
        // Check for Text.Format module patterns
        if fullName.Contains("Text.Format.intToString") || fullName.EndsWith(".intToString") then
            Some "intToString"
        elif fullName.Contains("Text.Format.int64ToString") || fullName.EndsWith(".int64ToString") then
            Some "int64ToString"
        else None
    | None -> None

/// Emit code to write an i32 integer value to stdout as decimal digits
/// This implements intToString inline by converting and writing directly
let emitWriteInt32 (builder: MLIRBuilder) (ctx: SSAContext) (valueReg: string) : unit =
    // Algorithm: Convert integer to decimal string and write to stdout
    // We'll allocate a 12-byte buffer (enough for -2147483648 + null)
    // Then fill it right-to-left with digits, then write

    let bufSize = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 12 : i64" bufSize)
    let buf = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x i8 : (i64) -> !llvm.ptr" buf bufSize)

    // We need to handle: negative sign, digits, then write from first digit
    // For simplicity, we'll emit a loop that divides by 10 and stores digits

    // Store digits right-to-left, track position
    let endPos = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 11 : i64" endPos)  // Last position
    let endPtr = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr, i8" endPtr buf endPos)

    // Store null terminator
    let nullChar = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 0 : i8" nullChar)
    MLIRBuilder.line builder (sprintf "llvm.store %s, %s : i8, !llvm.ptr" nullChar endPtr)

    // We'll use a simplified approach: just write the value with inline asm printf-style
    // Actually, for now emit a simple integer write using itoa-style loop

    // Check if negative
    let zero32 = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 0 : i32" zero32)
    let isNeg = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.cmpi slt, %s, %s : i32" isNeg valueReg zero32)

    // Get absolute value
    let negated = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.subi %s, %s : i32" negated zero32 valueReg)
    let absVal = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.select %s, %s, %s : i32" absVal isNeg negated valueReg)

    // Extend to i64 for arithmetic
    let val64 = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.extui %s : i32 to i64" val64 absVal)

    // Position counter (starts at 10, goes down)
    let posAlloc = SSAContext.nextValue ctx
    let allocOne = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i32" allocOne)
    MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x i64 : (i32) -> !llvm.ptr" posAlloc allocOne)
    let initPos = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 10 : i64" initPos)
    MLIRBuilder.line builder (sprintf "llvm.store %s, %s : i64, !llvm.ptr" initPos posAlloc)

    // Value counter
    let valAlloc = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x i64 : (i32) -> !llvm.ptr" valAlloc allocOne)
    MLIRBuilder.line builder (sprintf "llvm.store %s, %s : i64, !llvm.ptr" val64 valAlloc)

    // Loop to extract digits
    let loopLabel = sprintf "int_to_str_%d" ctx.Counter
    ctx.Counter <- ctx.Counter + 1
    let exitLabel = sprintf "int_to_str_done_%d" ctx.Counter
    ctx.Counter <- ctx.Counter + 1

    MLIRBuilder.line builder (sprintf "cf.br ^%s" loopLabel)
    MLIRBuilder.lineNoIndent builder (sprintf "^%s:" loopLabel)

    // Load current value
    let curVal = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.load %s : !llvm.ptr -> i64" curVal valAlloc)

    // Get digit: curVal % 10
    let ten64 = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 10 : i64" ten64)
    let digit = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.remui %s, %s : i64" digit curVal ten64)

    // Convert to ASCII: digit + '0'
    let ascii0 = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 48 : i64" ascii0)  // '0' = 48
    let asciiDigit = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.addi %s, %s : i64" asciiDigit digit ascii0)
    let asciiChar = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.trunci %s : i64 to i8" asciiChar asciiDigit)

    // Load position and store digit
    let curPos = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.load %s : !llvm.ptr -> i64" curPos posAlloc)
    let digitPtr = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr, i8" digitPtr buf curPos)
    MLIRBuilder.line builder (sprintf "llvm.store %s, %s : i8, !llvm.ptr" asciiChar digitPtr)

    // Decrement position
    let one64 = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i64" one64)
    let newPos = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.subi %s, %s : i64" newPos curPos one64)
    MLIRBuilder.line builder (sprintf "llvm.store %s, %s : i64, !llvm.ptr" newPos posAlloc)

    // Divide value by 10
    let newVal = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.divui %s, %s : i64" newVal curVal ten64)
    MLIRBuilder.line builder (sprintf "llvm.store %s, %s : i64, !llvm.ptr" newVal valAlloc)

    // Check if done (newVal == 0)
    let zero64 = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 0 : i64" zero64)
    let isDone = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.cmpi eq, %s, %s : i64" isDone newVal zero64)
    MLIRBuilder.line builder (sprintf "cf.cond_br %s, ^%s, ^%s" isDone exitLabel loopLabel)

    MLIRBuilder.lineNoIndent builder (sprintf "^%s:" exitLabel)

    // If negative, add '-' sign at current position
    let negLabel = sprintf "add_neg_%d" ctx.Counter
    ctx.Counter <- ctx.Counter + 1
    let writeLabel = sprintf "do_write_%d" ctx.Counter
    ctx.Counter <- ctx.Counter + 1

    MLIRBuilder.line builder (sprintf "cf.cond_br %s, ^%s, ^%s" isNeg negLabel writeLabel)

    MLIRBuilder.lineNoIndent builder (sprintf "^%s:" negLabel)
    let finalPos = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.load %s : !llvm.ptr -> i64" finalPos posAlloc)
    let minusPtr = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr, i8" minusPtr buf finalPos)
    let minusChar = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 45 : i8" minusChar)  // '-' = 45
    MLIRBuilder.line builder (sprintf "llvm.store %s, %s : i8, !llvm.ptr" minusChar minusPtr)
    let posMinusOne = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.subi %s, %s : i64" posMinusOne finalPos one64)
    MLIRBuilder.line builder (sprintf "llvm.store %s, %s : i64, !llvm.ptr" posMinusOne posAlloc)
    MLIRBuilder.line builder (sprintf "cf.br ^%s" writeLabel)

    MLIRBuilder.lineNoIndent builder (sprintf "^%s:" writeLabel)
    // Calculate start position and length
    let startPos = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.load %s : !llvm.ptr -> i64" startPos posAlloc)
    let startPosPlus1 = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.addi %s, %s : i64" startPosPlus1 startPos one64)
    let startPtr = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr, i8" startPtr buf startPosPlus1)

    // Length = 11 - (startPos + 1) = 10 - startPos
    let eleven = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 11 : i64" eleven)
    let length = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.subi %s, %s : i64" length eleven startPosPlus1)

    // Write syscall
    let fd = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i64" fd)
    let sysWrite = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i64" sysWrite)
    let writeResult = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"=r,{rax},{rdi},{rsi},{rdx}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64"
        writeResult sysWrite fd startPtr length)

/// Emit code to write an i64 integer value to stdout as decimal digits
let emitWriteInt64 (builder: MLIRBuilder) (ctx: SSAContext) (valueReg: string) : unit =
    // Similar to emitWriteInt32 but for 64-bit values
    // Buffer needs to be 21 bytes (enough for -9223372036854775808 + null)

    let bufSize = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 21 : i64" bufSize)
    let buf = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x i8 : (i64) -> !llvm.ptr" buf bufSize)

    // Store null terminator at end
    let endPos = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 20 : i64" endPos)
    let endPtr = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr, i8" endPtr buf endPos)
    let nullChar = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 0 : i8" nullChar)
    MLIRBuilder.line builder (sprintf "llvm.store %s, %s : i8, !llvm.ptr" nullChar endPtr)

    // Check if negative
    let zero64 = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 0 : i64" zero64)
    let isNeg = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.cmpi slt, %s, %s : i64" isNeg valueReg zero64)

    // Get absolute value
    let negated = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.subi %s, %s : i64" negated zero64 valueReg)
    let absVal = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.select %s, %s, %s : i64" absVal isNeg negated valueReg)

    // Position counter (starts at 19, goes down)
    let allocOne = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i32" allocOne)
    let posAlloc = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x i64 : (i32) -> !llvm.ptr" posAlloc allocOne)
    let initPos = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 19 : i64" initPos)
    MLIRBuilder.line builder (sprintf "llvm.store %s, %s : i64, !llvm.ptr" initPos posAlloc)

    // Value counter
    let valAlloc = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x i64 : (i32) -> !llvm.ptr" valAlloc allocOne)
    MLIRBuilder.line builder (sprintf "llvm.store %s, %s : i64, !llvm.ptr" absVal valAlloc)

    // Loop to extract digits
    let loopLabel = sprintf "i64_to_str_%d" ctx.Counter
    ctx.Counter <- ctx.Counter + 1
    let exitLabel = sprintf "i64_to_str_done_%d" ctx.Counter
    ctx.Counter <- ctx.Counter + 1

    MLIRBuilder.line builder (sprintf "cf.br ^%s" loopLabel)
    MLIRBuilder.lineNoIndent builder (sprintf "^%s:" loopLabel)

    let curVal = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.load %s : !llvm.ptr -> i64" curVal valAlloc)

    let ten64 = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 10 : i64" ten64)
    let digit = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.remui %s, %s : i64" digit curVal ten64)

    let ascii0 = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 48 : i64" ascii0)
    let asciiDigit = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.addi %s, %s : i64" asciiDigit digit ascii0)
    let asciiChar = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.trunci %s : i64 to i8" asciiChar asciiDigit)

    let curPos = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.load %s : !llvm.ptr -> i64" curPos posAlloc)
    let digitPtr = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr, i8" digitPtr buf curPos)
    MLIRBuilder.line builder (sprintf "llvm.store %s, %s : i8, !llvm.ptr" asciiChar digitPtr)

    let one64 = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i64" one64)
    let newPos = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.subi %s, %s : i64" newPos curPos one64)
    MLIRBuilder.line builder (sprintf "llvm.store %s, %s : i64, !llvm.ptr" newPos posAlloc)

    let newVal = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.divui %s, %s : i64" newVal curVal ten64)
    MLIRBuilder.line builder (sprintf "llvm.store %s, %s : i64, !llvm.ptr" newVal valAlloc)

    let isDone = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.cmpi eq, %s, %s : i64" isDone newVal zero64)
    MLIRBuilder.line builder (sprintf "cf.cond_br %s, ^%s, ^%s" isDone exitLabel loopLabel)

    MLIRBuilder.lineNoIndent builder (sprintf "^%s:" exitLabel)

    let negLabel = sprintf "add_neg64_%d" ctx.Counter
    ctx.Counter <- ctx.Counter + 1
    let writeLabel = sprintf "do_write64_%d" ctx.Counter
    ctx.Counter <- ctx.Counter + 1

    MLIRBuilder.line builder (sprintf "cf.cond_br %s, ^%s, ^%s" isNeg negLabel writeLabel)

    MLIRBuilder.lineNoIndent builder (sprintf "^%s:" negLabel)
    let finalPos = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.load %s : !llvm.ptr -> i64" finalPos posAlloc)
    let minusPtr = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr, i8" minusPtr buf finalPos)
    let minusChar = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 45 : i8" minusChar)
    MLIRBuilder.line builder (sprintf "llvm.store %s, %s : i8, !llvm.ptr" minusChar minusPtr)
    let posMinusOne = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.subi %s, %s : i64" posMinusOne finalPos one64)
    MLIRBuilder.line builder (sprintf "llvm.store %s, %s : i64, !llvm.ptr" posMinusOne posAlloc)
    MLIRBuilder.line builder (sprintf "cf.br ^%s" writeLabel)

    MLIRBuilder.lineNoIndent builder (sprintf "^%s:" writeLabel)
    let startPos = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.load %s : !llvm.ptr -> i64" startPos posAlloc)
    let startPosPlus1 = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.addi %s, %s : i64" startPosPlus1 startPos one64)
    let startPtr = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr, i8" startPtr buf startPosPlus1)

    let twenty = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 20 : i64" twenty)
    let length = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.subi %s, %s : i64" length twenty startPosPlus1)

    let fd = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i64" fd)
    let sysWrite = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i64" sysWrite)
    let writeResult = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"=r,{rax},{rdi},{rsi},{rdx}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64"
        writeResult sysWrite fd startPtr length)

/// Emit clock_gettime syscall for Linux and return ticks (100-nanosecond intervals)
let emitClockGettime (builder: MLIRBuilder) (ctx: SSAContext) (clockId: int) : string =
    // Allocate timespec on stack: struct { int64 tv_sec; int64 tv_nsec; }
    let one = SSAContext.nextValue ctx
    let timespec = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i32" one)
    MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x !llvm.struct<(i64, i64)> : (i32) -> !llvm.ptr" timespec one)

    // Prepare syscall arguments
    let clockIdVal = SSAContext.nextValue ctx
    let syscallNum = SSAContext.nextValue ctx
    let result = SSAContext.nextValue ctx

    MLIRBuilder.line builder (sprintf "%s = arith.constant %d : i64" clockIdVal clockId)
    MLIRBuilder.line builder (sprintf "%s = arith.constant 228 : i64" syscallNum)  // clock_gettime syscall number

    // Execute syscall: clock_gettime(clock_id, &timespec)
    MLIRBuilder.line builder (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"=r,{rax},{rdi},{rsi}\" %s, %s, %s : (i64, i64, !llvm.ptr) -> i64"
        result syscallNum clockIdVal timespec)

    // Extract seconds and nanoseconds from timespec
    let secPtr = SSAContext.nextValue ctx
    let nsecPtr = SSAContext.nextValue ctx
    let sec = SSAContext.nextValue ctx
    let nsec = SSAContext.nextValue ctx

    MLIRBuilder.line builder (sprintf "%s = llvm.getelementptr %s[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)>" secPtr timespec)
    MLIRBuilder.line builder (sprintf "%s = llvm.getelementptr %s[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)>" nsecPtr timespec)
    MLIRBuilder.line builder (sprintf "%s = llvm.load %s : !llvm.ptr -> i64" sec secPtr)
    MLIRBuilder.line builder (sprintf "%s = llvm.load %s : !llvm.ptr -> i64" nsec nsecPtr)

    // Convert to 100-nanosecond ticks (.NET DateTime ticks format)
    // ticks = sec * 10_000_000 + nsec / 100
    let ticksPerSec = SSAContext.nextValue ctx
    let secTicks = SSAContext.nextValue ctx
    let nsecDiv = SSAContext.nextValue ctx
    let nsecTicks = SSAContext.nextValue ctx
    let totalTicks = SSAContext.nextValue ctx

    MLIRBuilder.line builder (sprintf "%s = arith.constant 10000000 : i64" ticksPerSec)
    MLIRBuilder.line builder (sprintf "%s = arith.muli %s, %s : i64" secTicks sec ticksPerSec)
    MLIRBuilder.line builder (sprintf "%s = arith.constant 100 : i64" nsecDiv)
    MLIRBuilder.line builder (sprintf "%s = arith.divui %s, %s : i64" nsecTicks nsec nsecDiv)
    MLIRBuilder.line builder (sprintf "%s = arith.addi %s, %s : i64" totalTicks secTicks nsecTicks)

    // FIDELITY: Register the return type as i64
    SSAContext.registerSSAType ctx totalTicks "i64"
    totalTicks

/// Emit nanosleep syscall for Linux
let emitNanosleep (builder: MLIRBuilder) (ctx: SSAContext) (milliseconds: string) : unit =
    // Convert milliseconds to seconds and nanoseconds
    // seconds = ms / 1000
    // nanoseconds = (ms % 1000) * 1_000_000
    let thousand = SSAContext.nextValue ctx
    let million = SSAContext.nextValue ctx
    let msExtended = SSAContext.nextValue ctx
    let seconds = SSAContext.nextValue ctx
    let remainder = SSAContext.nextValue ctx
    let nanoseconds = SSAContext.nextValue ctx

    MLIRBuilder.line builder (sprintf "%s = arith.constant 1000 : i64" thousand)
    MLIRBuilder.line builder (sprintf "%s = arith.constant 1000000 : i64" million)
    MLIRBuilder.line builder (sprintf "%s = arith.extsi %s : i32 to i64" msExtended milliseconds)
    MLIRBuilder.line builder (sprintf "%s = arith.divui %s, %s : i64" seconds msExtended thousand)
    MLIRBuilder.line builder (sprintf "%s = arith.remui %s, %s : i64" remainder msExtended thousand)
    MLIRBuilder.line builder (sprintf "%s = arith.muli %s, %s : i64" nanoseconds remainder million)

    // Allocate request and remainder timespec structs
    let one = SSAContext.nextValue ctx
    let reqTimespec = SSAContext.nextValue ctx
    let remTimespec = SSAContext.nextValue ctx

    MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i32" one)
    MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x !llvm.struct<(i64, i64)> : (i32) -> !llvm.ptr" reqTimespec one)
    MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x !llvm.struct<(i64, i64)> : (i32) -> !llvm.ptr" remTimespec one)

    // Store seconds and nanoseconds into request timespec
    let secPtr = SSAContext.nextValue ctx
    let nsecPtr = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.getelementptr %s[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)>" secPtr reqTimespec)
    MLIRBuilder.line builder (sprintf "%s = llvm.getelementptr %s[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)>" nsecPtr reqTimespec)
    MLIRBuilder.line builder (sprintf "llvm.store %s, %s : i64, !llvm.ptr" seconds secPtr)
    MLIRBuilder.line builder (sprintf "llvm.store %s, %s : i64, !llvm.ptr" nanoseconds nsecPtr)

    // Call nanosleep syscall
    let syscallNum = SSAContext.nextValue ctx
    let result = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 35 : i64" syscallNum)  // nanosleep syscall number
    MLIRBuilder.line builder (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"=r,{rax},{rdi},{rsi}\" %s, %s, %s : (i64, !llvm.ptr, !llvm.ptr) -> i64"
        result syscallNum reqTimespec remTimespec)

/// Emit a Time operation for Linux x86-64
let emitTimeOperation (operation: string) (builder: MLIRBuilder) (ctx: SSAContext) (args: string list) : string option =
    match operation with
    | "currentTicks" ->
        let ticks = emitClockGettime builder ctx 0  // CLOCK_REALTIME
        // Add Unix epoch offset to convert to .NET ticks
        let epochOffset = SSAContext.nextValue ctx
        let result = SSAContext.nextValue ctx
        MLIRBuilder.line builder (sprintf "%s = arith.constant 621355968000000000 : i64" epochOffset)
        MLIRBuilder.line builder (sprintf "%s = arith.addi %s, %s : i64" result ticks epochOffset)
        // FIDELITY: Register the return type as i64
        SSAContext.registerSSAType ctx result "i64"
        Some result

    | "highResolutionTicks" ->
        let ticks = emitClockGettime builder ctx 1  // CLOCK_MONOTONIC
        // ticks already has type registered via emitClockGettime
        Some ticks

    | "tickFrequency" ->
        // Linux clock_gettime uses nanoseconds, converted to 100ns ticks
        // Frequency is 10,000,000 ticks per second
        let freq = SSAContext.nextValue ctx
        MLIRBuilder.line builder (sprintf "%s = arith.constant 10000000 : i64" freq)
        // FIDELITY: Register the return type as i64
        SSAContext.registerSSAType ctx freq "i64"
        Some freq

    | "currentUnixTimestamp" ->
        // Get Unix timestamp (seconds since epoch) directly from clock_gettime
        // Allocate timespec on stack: struct { int64 tv_sec; int64 tv_nsec; }
        let one = SSAContext.nextValue ctx
        let timespec = SSAContext.nextValue ctx
        MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i32" one)
        MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x !llvm.struct<(i64, i64)> : (i32) -> !llvm.ptr" timespec one)

        // Call clock_gettime with CLOCK_REALTIME (0)
        let clockIdVal = SSAContext.nextValue ctx
        let syscallNum = SSAContext.nextValue ctx
        let syscallResult = SSAContext.nextValue ctx
        MLIRBuilder.line builder (sprintf "%s = arith.constant 0 : i64" clockIdVal)
        MLIRBuilder.line builder (sprintf "%s = arith.constant 228 : i64" syscallNum)
        MLIRBuilder.line builder (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"=r,{rax},{rdi},{rsi}\" %s, %s, %s : (i64, i64, !llvm.ptr) -> i64"
            syscallResult syscallNum clockIdVal timespec)

        // Extract just the seconds (first field of timespec)
        let secPtr = SSAContext.nextValue ctx
        let seconds = SSAContext.nextValue ctx
        MLIRBuilder.line builder (sprintf "%s = llvm.getelementptr %s[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)>" secPtr timespec)
        MLIRBuilder.line builder (sprintf "%s = llvm.load %s : !llvm.ptr -> i64" seconds secPtr)

        // FIDELITY: Register the return type as i64
        SSAContext.registerSSAType ctx seconds "i64"
        Some seconds

    | "sleep" ->
        match args with
        | [msArg] ->
            emitNanosleep builder ctx msArg
            Some ""  // Void return
        | _ ->
            MLIRBuilder.line builder "// sleep requires milliseconds argument"
            None

    | _ -> None

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

/// Walk children of a node (already in source order)
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

/// Check if a function binding should be emitted as a separate function
/// (vs being inlined at call site like Time operations and Text.Format operations)
let shouldEmitAsFunction (node: PSGNode) : bool =
    match node.Symbol with
    | Some sym ->
        let fullName = sym.FullName
        // Skip operations that are inlined at call sites
        not (// Time operations
             fullName.Contains("Time.") ||
             fullName.Contains("TimeApi.") ||
             fullName.EndsWith(".highResolutionTicks") ||
             fullName.EndsWith(".tickFrequency") ||
             fullName.EndsWith(".currentTicks") ||
             fullName.EndsWith(".currentUnixTimestamp") ||
             fullName.EndsWith(".sleep") ||
             // Text.Format operations - inlined at Console.Write/WriteLine call sites
             fullName.Contains("Text.Format.") ||
             fullName.EndsWith(".intToString") ||
             fullName.EndsWith(".int64ToString") ||
             // Console operations - inlined
             fullName.Contains("Console.") ||
             fullName.EndsWith(".Write") ||
             fullName.EndsWith(".WriteLine"))
    | None -> true

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
    |> List.filter shouldEmitAsFunction

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

    | sk when sk.StartsWith("Const:Int32 ") ->
        // Format: "Const:Int32 5" (space separated)
        let value = sk.Substring("Const:Int32 ".Length)
        let result = SSAContext.nextValue ctx
        MLIRBuilder.line builder (sprintf "%s = arith.constant %s : i32" result value)
        SSAContext.registerSSAType ctx result "i32"
        Some result

    | sk when sk.StartsWith("Const:Int64:") ->
        // Int64 constant - Format: "Const:Int64:1000"
        let value = sk.Substring("Const:Int64:".Length).TrimEnd('L')
        let result = SSAContext.nextValue ctx
        MLIRBuilder.line builder (sprintf "%s = arith.constant %s : i64" result value)
        SSAContext.registerSSAType ctx result "i64"
        Some result

    | sk when sk.StartsWith("Const:Int64 ") ->
        // Format: "Const:Int64 1000L" or "Const:Int64 1000"
        let rawValue = sk.Substring("Const:Int64 ".Length)
        let value = rawValue.TrimEnd('L')
        let result = SSAContext.nextValue ctx
        MLIRBuilder.line builder (sprintf "%s = arith.constant %s : i64" result value)
        SSAContext.registerSSAType ctx result "i64"
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

        // Check if this is a Time operation
        let timeOp =
            match callTarget with
            | Some target -> isTimeOperation target
            | None -> None

        // Check if this is an arithmetic operator
        // Check both Symbol.DisplayName and SyntaxKind for operator detection
        // Returns (opName, typeHint) - typeHint may be None if type needs to be inferred from operands
        let arithmeticOp =
            // Helper to detect arithmetic op from a node
            let detectArithOp (n: PSGNode) =
                // Try to get type hint from the node's type information
                let typeHint =
                    match n.Type with
                    | Some ftype ->
                        try
                            if ftype.HasTypeDefinition then
                                match ftype.TypeDefinition.TryFullName with
                                | Some "System.Int64" -> Some "i64"
                                | Some "System.Int32" -> Some "i32"
                                | _ -> None
                            else None
                        with _ -> None
                    | None -> None

                let fromSymbol =
                    match n.Symbol with
                    | Some sym ->
                        // Symbol.DisplayName can be either "op_Addition", "(+)", or Alloy.Numerics style like "Multiply"
                        let dispName = sym.DisplayName
                        let fullName = sym.FullName
                        match dispName with
                        | "op_Subtraction" | "(-)" | "Subtract" -> Some ("subi", typeHint)
                        | "op_Addition" | "(+)" | "Add" -> Some ("addi", typeHint)
                        | "op_Multiply" | "(*)" | "Multiply" -> Some ("muli", typeHint)
                        | "op_Division" | "(/)" | "Divide" -> Some ("divsi", typeHint)
                        | "op_Modulus" | "(%)" | "Modulus" -> Some ("remsi", typeHint)
                        | _ ->
                            // Also check FullName for Alloy.Numerics.*
                            if fullName.EndsWith(".Multiply") then Some ("muli", typeHint)
                            elif fullName.EndsWith(".Divide") then Some ("divsi", typeHint)
                            elif fullName.EndsWith(".Add") then Some ("addi", typeHint)
                            elif fullName.EndsWith(".Subtract") then Some ("subi", typeHint)
                            elif fullName.EndsWith(".Modulus") then Some ("remsi", typeHint)
                            else None
                    | None -> None

                match fromSymbol with
                | Some op -> Some op
                | None ->
                    // Check SyntaxKind for both standard F# operators and Alloy.Numerics operators
                    if n.SyntaxKind.Contains("op_Subtraction") || n.SyntaxKind.Contains("Subtract") then Some ("subi", typeHint)
                    elif n.SyntaxKind.Contains("op_Addition") || n.SyntaxKind.Contains("Add") then Some ("addi", typeHint)
                    elif n.SyntaxKind.Contains("op_Multiply") || n.SyntaxKind.Contains("Multiply") then Some ("muli", typeHint)
                    elif n.SyntaxKind.Contains("op_Division") || n.SyntaxKind.Contains("Divide") then Some ("divsi", typeHint)
                    elif n.SyntaxKind.Contains("op_Modulus") || n.SyntaxKind.Contains("Modulus") then Some ("remsi", typeHint)
                    else None

            // First check callTarget directly
            let direct =
                match callTarget with
                | Some target -> detectArithOp target
                | None -> None

            // If not found, check children and grandchildren (for nested App structures)
            match direct with
            | Some _ -> direct
            | None ->
                children
                |> List.tryPick (fun child ->
                    match detectArithOp child with
                    | Some op -> Some op
                    | None ->
                        // Also check grandchildren (for nested App)
                        let grandChildren = getChildren psg child
                        grandChildren |> List.tryPick detectArithOp)

        // Check if this is a comparison operator (need to look in children too for curried ops)
        let comparisonOp =
            // Helper to detect comparison op from a node
            let detectCompOp (n: PSGNode) =
                let fromSymbol =
                    match n.Symbol with
                    | Some sym ->
                        // Symbol.DisplayName can be either "op_LessThan" or "(<)" style
                        match sym.DisplayName with
                        | "op_LessThan" | "(<)" -> Some "slt"
                        | "op_GreaterThan" | "(>)" -> Some "sgt"
                        | "op_LessThanOrEqual" | "(<=)" -> Some "sle"
                        | "op_GreaterThanOrEqual" | "(>=)" -> Some "sge"
                        | "op_Equality" | "(=)" -> Some "eq"
                        | "op_Inequality" | "(<>)" -> Some "ne"
                        | _ -> None
                    | None -> None

                match fromSymbol with
                | Some op -> Some op
                | None ->
                    if n.SyntaxKind.Contains("op_LessThan") then Some "slt"
                    elif n.SyntaxKind.Contains("op_GreaterThan") then Some "sgt"
                    elif n.SyntaxKind.Contains("op_LessThanOrEqual") then Some "sle"
                    elif n.SyntaxKind.Contains("op_GreaterThanOrEqual") then Some "sge"
                    elif n.SyntaxKind.Contains("op_Equality") then Some "eq"
                    elif n.SyntaxKind.Contains("op_Inequality") then Some "ne"
                    else None

            // First check callTarget directly
            let direct =
                match callTarget with
                | Some target -> detectCompOp target
                | None -> None

            // If not found, check all children for operator
            match direct with
            | Some _ -> direct
            | None ->
                children
                |> List.tryPick (fun child ->
                    match detectCompOp child with
                    | Some op -> Some op
                    | None ->
                        // Also check grandchildren (for nested App)
                        let grandChildren = getChildren psg child
                        grandChildren |> List.tryPick detectCompOp)

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
                    // Check if argument is intToString or int64ToString call
                    let rec findIntToStringCall nodes =
                        match nodes with
                        | [] -> None
                        | node :: rest ->
                            if node.SyntaxKind.StartsWith("App") then
                                let appChildren = getChildren psg node
                                let targetNode = appChildren |> List.tryHead
                                match targetNode with
                                | Some target ->
                                    match isTextFormatOperation target with
                                    | Some "intToString" ->
                                        let argNode = appChildren |> List.tryItem 1
                                        Some ("intToString", argNode)
                                    | Some "int64ToString" ->
                                        let argNode = appChildren |> List.tryItem 1
                                        Some ("int64ToString", argNode)
                                    | _ ->
                                        match findIntToStringCall appChildren with
                                        | Some r -> Some r
                                        | None -> findIntToStringCall rest
                                | None -> findIntToStringCall rest
                            else
                                let nodeChildren = getChildren psg node
                                match findIntToStringCall nodeChildren with
                                | Some r -> Some r
                                | None -> findIntToStringCall rest

                    match findIntToStringCall children with
                    | Some ("intToString", Some argNode) ->
                        MLIRBuilder.line builder "// Console.WriteLine(intToString(expr))"
                        match emitExpression builder ctx psg argNode with
                        | Some valueReg ->
                            emitWriteInt32 builder ctx valueReg
                            // Also write newline
                            emitConsoleWrite builder ctx "\n"
                        | None ->
                            MLIRBuilder.line builder "// intToString argument evaluation failed"
                    | Some ("int64ToString", Some argNode) ->
                        MLIRBuilder.line builder "// Console.WriteLine(int64ToString(expr))"
                        match emitExpression builder ctx psg argNode with
                        | Some valueReg ->
                            emitWriteInt64 builder ctx valueReg
                            emitConsoleWrite builder ctx "\n"
                        | None ->
                            MLIRBuilder.line builder "// int64ToString argument evaluation failed"
                    | _ ->
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
                    // Check if argument is intToString or int64ToString call
                    // Look for App nodes with intToString/int64ToString as the function
                    let rec findIntToStringCall nodes =
                        match nodes with
                        | [] -> None
                        | node :: rest ->
                            // Check if this is an App with intToString target
                            if node.SyntaxKind.StartsWith("App") then
                                let appChildren = getChildren psg node
                                let targetNode = appChildren |> List.tryHead
                                match targetNode with
                                | Some target ->
                                    match isTextFormatOperation target with
                                    | Some "intToString" ->
                                        // Found intToString - get the argument
                                        let argNode = appChildren |> List.tryItem 1
                                        Some ("intToString", argNode)
                                    | Some "int64ToString" ->
                                        let argNode = appChildren |> List.tryItem 1
                                        Some ("int64ToString", argNode)
                                    | _ ->
                                        // Check children recursively
                                        match findIntToStringCall appChildren with
                                        | Some r -> Some r
                                        | None -> findIntToStringCall rest
                                | None -> findIntToStringCall rest
                            else
                                let nodeChildren = getChildren psg node
                                match findIntToStringCall nodeChildren with
                                | Some r -> Some r
                                | None -> findIntToStringCall rest

                    match findIntToStringCall children with
                    | Some ("intToString", Some argNode) ->
                        // Emit the argument expression and write it as integer
                        MLIRBuilder.line builder "// Console.Write(intToString(expr))"
                        match emitExpression builder ctx psg argNode with
                        | Some valueReg ->
                            emitWriteInt32 builder ctx valueReg
                        | None ->
                            MLIRBuilder.line builder "// intToString argument evaluation failed"
                    | Some ("int64ToString", Some argNode) ->
                        // Emit the argument expression and write it as 64-bit integer
                        MLIRBuilder.line builder "// Console.Write(int64ToString(expr))"
                        match emitExpression builder ctx psg argNode with
                        | Some valueReg ->
                            emitWriteInt64 builder ctx valueReg
                        | None ->
                            MLIRBuilder.line builder "// int64ToString argument evaluation failed"
                    | _ ->
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

        elif timeOp.IsSome then
            // Time operation: highResolutionTicks, tickFrequency, etc.
            let operation = timeOp.Value
            MLIRBuilder.line builder (sprintf "// Time operation: %s" operation)

            // For sleep, we need to get the milliseconds argument
            if operation = "sleep" then
                // Find the milliseconds argument
                let msArg = children |> List.tryFind (fun c ->
                    c.SyntaxKind.StartsWith("Const:Int") ||
                    c.SyntaxKind.StartsWith("Ident"))

                match msArg with
                | Some arg ->
                    let msValue =
                        if arg.SyntaxKind.StartsWith("Const:Int32:") then
                            // Constant value
                            let value = arg.SyntaxKind.Substring("Const:Int32:".Length)
                            let result = SSAContext.nextValue ctx
                            MLIRBuilder.line builder (sprintf "%s = arith.constant %s : i32" result value)
                            result
                        else
                            // Variable reference
                            match arg.Symbol with
                            | Some sym ->
                                match SSAContext.lookupLocal ctx sym.DisplayName with
                                | Some ssaName -> ssaName
                                | None ->
                                    let result = SSAContext.nextValue ctx
                                    MLIRBuilder.line builder (sprintf "%s = arith.constant 1000 : i32" result)  // Default 1 second
                                    result
                            | None ->
                                let result = SSAContext.nextValue ctx
                                MLIRBuilder.line builder (sprintf "%s = arith.constant 1000 : i32" result)
                                result
                    emitTimeOperation "sleep" builder ctx [msValue]
                | None ->
                    // Default to 1 second sleep
                    let msValue = SSAContext.nextValue ctx
                    MLIRBuilder.line builder (sprintf "%s = arith.constant 1000 : i32" msValue)
                    emitTimeOperation "sleep" builder ctx [msValue]
            else
                emitTimeOperation operation builder ctx []

        elif comparisonOp.IsSome then
            // Comparison operation: op_LessThan, op_GreaterThan, etc.
            // Structure is App(App(op, lhs), rhs) - curried binary operator
            let cmpOp = comparisonOp.Value

            // Collect operands from nested curried structure
            // For App(App(op, lhs), rhs): first child is App(op, lhs), second is rhs
            let rec collectOperands nodes acc =
                match nodes with
                | [] -> acc
                | node :: rest ->
                    // Check if this is an operator
                    // Symbol DisplayName can be "(<)" or "op_LessThan" style
                    let isOp =
                        match node.Symbol with
                        | Some sym ->
                            let dispName = sym.DisplayName
                            dispName.StartsWith("op_") ||
                            (dispName.StartsWith("(") && dispName.EndsWith(")")) // "(<)" style operators
                        | None -> node.SyntaxKind.Contains("op_")

                    if isOp then
                        collectOperands rest acc
                    elif node.SyntaxKind.StartsWith("App") then
                        // Nested App - recurse into it
                        let innerChildren = getChildren psg node
                        collectOperands (innerChildren @ rest) acc
                    else
                        // This is an operand
                        collectOperands rest (node :: acc)

            let operandNodes = collectOperands children [] |> List.rev

            // Emit operands
            let operandValues = operandNodes |> List.choose (fun operand ->
                emitExpression builder ctx psg operand)

            match operandValues with
            | [lhs; rhs] ->
                let result = SSAContext.nextValue ctx
                MLIRBuilder.line builder (sprintf "%s = arith.cmpi %s, %s, %s : i32" result cmpOp lhs rhs)
                Some result
            | [single] ->
                // Partial application - return operand for outer App
                Some single
            | _ ->
                MLIRBuilder.line builder (sprintf "// Comparison op %s with unexpected operand count: %d (collected: %d nodes)" cmpOp operandValues.Length operandNodes.Length)
                None

        elif arithmeticOp.IsSome then
            // Arithmetic operation: op_Subtraction, op_Addition, etc.
            // Structure is App(App(op, lhs), rhs) - curried binary operator
            let (op, typeHint) = arithmeticOp.Value

            // Collect operands from nested curried structure
            // IMPORTANT: Don't recurse into nested Apps that are themselves arithmetic operations
            // Those should be evaluated as subexpressions, not flattened
            let rec collectOperands nodes acc =
                match nodes with
                | [] -> acc
                | node :: rest ->
                    // Check if this is an operator (the current operation's operator)
                    // Symbol DisplayName can be "(+)" or "op_Addition" style
                    let isCurrentOp =
                        match node.Symbol with
                        | Some sym ->
                            let dispName = sym.DisplayName
                            dispName.StartsWith("op_") ||
                            (dispName.StartsWith("(") && dispName.EndsWith(")")) // "(+)" style operators
                        | None -> node.SyntaxKind.Contains("op_")

                    if isCurrentOp then
                        collectOperands rest acc
                    elif node.SyntaxKind.StartsWith("App") then
                        // Check if this nested App is itself an arithmetic operation
                        // If so, treat it as an operand (subexpression) rather than flattening
                        let innerChildren = getChildren psg node
                        let hasNestedArithOp =
                            innerChildren |> List.exists (fun child ->
                                match child.Symbol with
                                | Some sym ->
                                    let dispName = sym.DisplayName
                                    (dispName.StartsWith("op_") &&
                                     (dispName.Contains("Addition") || dispName.Contains("Subtraction") ||
                                      dispName.Contains("Multiply") || dispName.Contains("Division") ||
                                      dispName.Contains("Modulus"))) ||
                                    (dispName = "(+)" || dispName = "(-)" || dispName = "(*)" ||
                                     dispName = "(/)" || dispName = "(%)")
                                | None ->
                                    child.SyntaxKind.Contains("op_Addition") ||
                                    child.SyntaxKind.Contains("op_Subtraction") ||
                                    child.SyntaxKind.Contains("op_Multiply") ||
                                    child.SyntaxKind.Contains("op_Division") ||
                                    child.SyntaxKind.Contains("op_Modulus"))

                        if hasNestedArithOp then
                            // This is a subexpression - treat as operand, don't flatten
                            collectOperands rest (node :: acc)
                        else
                            // This is curried application structure - recurse into it
                            collectOperands (innerChildren @ rest) acc
                    else
                        // This is an operand
                        collectOperands rest (node :: acc)

            let operandNodes = collectOperands children [] |> List.rev

            // Determine type from operand nodes if typeHint is None - FIDELITY: use preserved type info
            let inferTypeFromOperand (n: PSGNode) : string option =
                // First, try to get type from the PSG node's Type field
                match n.Type with
                | Some ftype ->
                    try
                        if ftype.HasTypeDefinition then
                            match ftype.TypeDefinition.TryFullName with
                            | Some "System.Int64" -> Some "i64"
                            | Some "System.Int32" -> Some "i32"
                            | _ -> None
                        else None
                    with _ -> None
                | None ->
                    // Try to look up type from registered local variable (FIDELITY)
                    match n.Symbol with
                    | Some sym ->
                        SSAContext.lookupLocalType ctx sym.DisplayName
                    | None ->
                        // Also try extracting name from SyntaxKind like "Ident:startTicks"
                        if n.SyntaxKind.StartsWith("Ident:") then
                            let name = n.SyntaxKind.Substring(6)
                            SSAContext.lookupLocalType ctx name
                        else None

            let inferredType =
                match typeHint with
                | Some t -> t  // Use type hint if available
                | None ->
                    // Try to infer from operand nodes
                    operandNodes
                    |> List.tryPick inferTypeFromOperand
                    |> Option.defaultValue "i32"  // Default to i32

            // Emit both operands
            let operandValues = operandNodes |> List.choose (fun operand ->
                emitExpression builder ctx psg operand)

            match operandValues with
            | [lhs; rhs] ->
                // FIDELITY: Also check SSA types of operands for type inference
                let operandSSAType =
                    [lhs; rhs]
                    |> List.tryPick (fun ssa -> SSAContext.lookupSSAType ctx ssa)

                let finalType =
                    match operandSSAType with
                    | Some t -> t  // Use type from operand SSA values
                    | None -> inferredType

                let result = SSAContext.nextValue ctx
                MLIRBuilder.line builder (sprintf "%s = arith.%s %s, %s : %s" result op lhs rhs finalType)
                // FIDELITY: Register the result type
                SSAContext.registerSSAType ctx result finalType
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

            // Check if first child is a function identifier (not to be emitted, just extract name)
            // vs an inner App (like for curried calls or partial application)
            let firstChild = children |> List.tryHead
            let firstChildIsFuncIdent =
                match firstChild with
                | Some fc -> fc.SyntaxKind.StartsWith("Ident:")
                | None -> false

            // Only emit first child if it's NOT a simple function identifier
            // (For function identifiers, we'll handle the call in the App handler itself)
            let firstChildResult =
                if firstChildIsFuncIdent then
                    None  // Don't emit - we'll handle call below
                else
                    match firstChild with
                    | Some fc -> emitExpression builder ctx psg fc
                    | None -> None

            // Debug: show what we have
            let childDescs = children |> List.map (fun c -> c.SyntaxKind) |> String.concat ", "
            MLIRBuilder.line builder (sprintf "// Outer App children (%d): [%s]" children.Length childDescs)
            MLIRBuilder.line builder (sprintf "// firstChildResult = %A, firstChildIsFuncIdent = %b" firstChildResult firstChildIsFuncIdent)

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
                    // General function call - collect arguments from children
                    let emitFunctionCallWithArgs funcName =
                        // Collect argument nodes (skip function name identifier)
                        let argNodes = children |> List.filter (fun c ->
                            not (c.SyntaxKind.StartsWith("Ident:") && c.SyntaxKind.Contains(funcName: string)) &&
                            not (c.SyntaxKind = "Ident"))

                        // Emit argument expressions
                        let argValues = argNodes |> List.choose (fun argNode ->
                            emitExpression builder ctx psg argNode)

                        let result = SSAContext.nextValue ctx
                        if argValues.IsEmpty then
                            MLIRBuilder.line builder (sprintf "%s = func.call @%s() : () -> i32" result funcName)
                        else
                            // Build argument list with types (assume i32 for now, FIDELITY TODO: use tracked types)
                            let argStr = argValues |> String.concat ", "
                            let typeStr = argValues |> List.map (fun v ->
                                SSAContext.lookupSSAType ctx v |> Option.defaultValue "i32") |> String.concat ", "
                            MLIRBuilder.line builder (sprintf "%s = func.call @%s(%s) : (%s) -> i32" result funcName argStr typeStr)
                        Some result

                    // First, try to get function name from first child if it's a function identifier
                    let funcNameFromFirstChild =
                        if firstChildIsFuncIdent then
                            match firstChild with
                            | Some fc when fc.SyntaxKind.StartsWith("Ident:") ->
                                Some (fc.SyntaxKind.Substring(6))  // Extract name after "Ident:"
                            | _ -> None
                        else None

                    // Helper to extract string content from Const:String node
                    let extractStringFromNode (n: PSGNode) =
                        let sk = n.SyntaxKind
                        let startQuote = sk.IndexOf("(\"")
                        if startQuote >= 0 then
                            let contentStart = startQuote + 2
                            let endQuote = sk.IndexOf("\",", contentStart)
                            if endQuote >= contentStart then  // Allow empty strings (endQuote == contentStart)
                                Some (sk.Substring(contentStart, endQuote - contentStart))
                            else None
                        else None

                    match funcNameFromFirstChild with
                    | Some "Write" ->
                        // Console.Write - handle string literal or intToString/int64ToString
                        let argNode = children |> List.tryItem 1
                        match argNode with
                        | Some arg when arg.SyntaxKind.StartsWith("Const:String") ->
                            match extractStringFromNode arg with
                            | Some content ->
                                emitConsoleWrite builder ctx content
                            | None ->
                                MLIRBuilder.line builder (sprintf "// Write: could not extract string from %s" arg.SyntaxKind)
                        | Some arg when arg.SyntaxKind.StartsWith("App") ->
                            // Could be intToString or int64ToString
                            let appChildren = getChildren psg arg
                            let funcTarget = appChildren |> List.tryHead
                            match funcTarget with
                            | Some ft when ft.SyntaxKind = "Ident:intToString" ->
                                let innerArg = appChildren |> List.tryItem 1
                                match innerArg with
                                | Some ia ->
                                    match emitExpression builder ctx psg ia with
                                    | Some valueReg ->
                                        emitWriteInt32 builder ctx valueReg
                                    | None ->
                                        MLIRBuilder.line builder "// intToString arg evaluation failed"
                                | None ->
                                    MLIRBuilder.line builder "// intToString missing argument"
                            | Some ft when ft.SyntaxKind = "Ident:int64ToString" ->
                                let innerArg = appChildren |> List.tryItem 1
                                match innerArg with
                                | Some ia ->
                                    match emitExpression builder ctx psg ia with
                                    | Some valueReg ->
                                        emitWriteInt64 builder ctx valueReg
                                    | None ->
                                        MLIRBuilder.line builder "// int64ToString arg evaluation failed"
                                | None ->
                                    MLIRBuilder.line builder "// int64ToString missing argument"
                            | _ ->
                                // Unknown App - try emitting and passing result
                                match emitExpression builder ctx psg arg with
                                | Some valueReg ->
                                    emitWriteInt32 builder ctx valueReg
                                | None ->
                                    MLIRBuilder.line builder "// Write: could not emit App argument"
                        | _ ->
                            MLIRBuilder.line builder "// Write: unhandled argument type"
                        None
                    | Some "WriteLine" ->
                        // Console.WriteLine - handle string literal or intToString/int64ToString
                        let argNode = children |> List.tryItem 1
                        match argNode with
                        | Some arg when arg.SyntaxKind.StartsWith("Const:String") ->
                            match extractStringFromNode arg with
                            | Some content ->
                                emitConsoleWriteLine builder ctx content
                            | None ->
                                MLIRBuilder.line builder (sprintf "// WriteLine: could not extract string from %s" arg.SyntaxKind)
                        | Some arg when arg.SyntaxKind.StartsWith("App") ->
                            // Could be intToString or int64ToString
                            let appChildren = getChildren psg arg
                            let funcTarget = appChildren |> List.tryHead
                            match funcTarget with
                            | Some ft when ft.SyntaxKind = "Ident:intToString" ->
                                let innerArg = appChildren |> List.tryItem 1
                                match innerArg with
                                | Some ia ->
                                    match emitExpression builder ctx psg ia with
                                    | Some valueReg ->
                                        emitWriteInt32 builder ctx valueReg
                                        emitConsoleWrite builder ctx "\n"
                                    | None ->
                                        MLIRBuilder.line builder "// intToString arg evaluation failed"
                                | None ->
                                    MLIRBuilder.line builder "// intToString missing argument"
                            | Some ft when ft.SyntaxKind = "Ident:int64ToString" ->
                                let innerArg = appChildren |> List.tryItem 1
                                match innerArg with
                                | Some ia ->
                                    match emitExpression builder ctx psg ia with
                                    | Some valueReg ->
                                        emitWriteInt64 builder ctx valueReg
                                        emitConsoleWrite builder ctx "\n"
                                    | None ->
                                        MLIRBuilder.line builder "// int64ToString arg evaluation failed"
                                | None ->
                                    MLIRBuilder.line builder "// int64ToString missing argument"
                            | _ ->
                                // Unknown App - try emitting and passing result
                                match emitExpression builder ctx psg arg with
                                | Some valueReg ->
                                    emitWriteInt32 builder ctx valueReg
                                    emitConsoleWrite builder ctx "\n"
                                | None ->
                                    MLIRBuilder.line builder "// WriteLine: could not emit App argument"
                        | _ ->
                            MLIRBuilder.line builder "// WriteLine: unhandled argument type"
                        None
                    | Some "sleep" ->
                        // Console.sleep ms - emit nanosleep syscall
                        let argNode = children |> List.tryItem 1
                        match argNode with
                        | Some arg ->
                            match emitExpression builder ctx psg arg with
                            | Some msReg ->
                                emitNanosleep builder ctx msReg
                                None
                            | None ->
                                MLIRBuilder.line builder "// sleep: could not emit ms argument"
                                None
                        | None ->
                            MLIRBuilder.line builder "// sleep: missing ms argument"
                            None
                    | Some "intToString" | Some "int64ToString" ->
                        // These should be handled at the call site (Write/WriteLine)
                        // If we get here, just emit the argument and return it
                        let argNode = children |> List.tryItem 1
                        match argNode with
                        | Some arg -> emitExpression builder ctx psg arg
                        | None -> None
                    | Some "highResolutionTicks" ->
                        // Time.highResolutionTicks() - emit clock_gettime
                        emitTimeOperation "highResolutionTicks" builder ctx []
                    | Some "tickFrequency" ->
                        // Time.tickFrequency() - return constant
                        emitTimeOperation "tickFrequency" builder ctx []
                    | Some "currentUnixTimestamp" ->
                        // Time.currentUnixTimestamp() - emit clock_gettime
                        emitTimeOperation "currentUnixTimestamp" builder ctx []
                    | Some "currentTicks" ->
                        // Time.currentTicks() - emit clock_gettime
                        emitTimeOperation "currentTicks" builder ctx []
                    | Some funcName ->
                        // General function call
                        emitFunctionCallWithArgs funcName
                    | None ->
                        // Fall back to checking node symbol or call target
                        match node.Symbol with
                        | Some sym ->
                            match sym with
                            | :? FSharpEntity as entity when entity.IsFSharpModule ->
                                // Skip module references
                                None
                            | :? FSharpMemberOrFunctionOrValue as mfv ->
                                let funcName = sym.DisplayName.Replace(".", "_")
                                emitFunctionCallWithArgs funcName
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
                                        emitFunctionCallWithArgs funcName
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
            // Check if this is a mutable variable (has _ptr registered)
            match SSAContext.lookupLocal ctx (name + "_ptr") with
            | Some ptrName ->
                // Load current value from mutable variable - FIDELITY: use correct type
                let varType = SSAContext.lookupLocalType ctx name |> Option.defaultValue "i32"
                let result = SSAContext.nextValue ctx
                MLIRBuilder.line builder (sprintf "%s = llvm.load %s : !llvm.ptr -> %s" result ptrName varType)
                Some result
            | None ->
                // Check for immutable local
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

        // Check if this is a mutable binding (check symbol for IsMutable)
        let isMutable =
            match patternNode with
            | Some pat ->
                match pat.Symbol with
                | Some sym ->
                    match sym with
                    | :? FSharpMemberOrFunctionOrValue as mfv -> mfv.IsMutable
                    | _ -> false
                | None -> false
            | None -> false

        // Extract type information from the pattern node - FIDELITY: preserve types through compilation
        let varType =
            match patternNode with
            | Some pat ->
                match pat.Type with
                | Some ftype ->
                    try
                        if ftype.HasTypeDefinition then
                            match ftype.TypeDefinition.TryFullName with
                            | Some "System.Int64" -> "i64"
                            | Some "System.Int32" -> "i32"
                            | Some "System.Boolean" -> "i1"
                            | Some "System.String" -> "!llvm.ptr"
                            | _ -> "i32"  // Default
                        else "i32"
                    with _ -> "i32"
                | None ->
                    // Try to get type from the expression node
                    match exprNode with
                    | Some expr ->
                        match expr.Type with
                        | Some ftype ->
                            try
                                if ftype.HasTypeDefinition then
                                    match ftype.TypeDefinition.TryFullName with
                                    | Some "System.Int64" -> "i64"
                                    | Some "System.Int32" -> "i32"
                                    | Some "System.Boolean" -> "i1"
                                    | Some "System.String" -> "!llvm.ptr"
                                    | _ -> "i32"
                                else "i32"
                            with _ -> "i32"
                        | None -> "i32"
                    | None -> "i32"
            | None -> "i32"

        // For mutable variables, allocate stack space
        if isMutable then
            match varName with
            | Some name ->
                // Allocate stack slot for mutable variable - FIDELITY: use correct type
                let llvmAllocType = if varType = "i64" then "i64" else "i32"
                let one = SSAContext.nextValue ctx
                let ptr = SSAContext.nextValue ctx
                MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i32" one)
                MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x %s : (i32) -> !llvm.ptr" ptr one llvmAllocType)

                // Store initial value
                match exprNode with
                | Some expr ->
                    match emitExpression builder ctx psg expr with
                    | Some initVal ->
                        MLIRBuilder.line builder (sprintf "llvm.store %s, %s : %s, !llvm.ptr" initVal ptr varType)
                    | None ->
                        // Default to 0
                        let zero = SSAContext.nextValue ctx
                        MLIRBuilder.line builder (sprintf "%s = arith.constant 0 : %s" zero varType)
                        MLIRBuilder.line builder (sprintf "llvm.store %s, %s : %s, !llvm.ptr" zero ptr varType)
                | None ->
                    let zero = SSAContext.nextValue ctx
                    MLIRBuilder.line builder (sprintf "%s = arith.constant 0 : %s" zero varType)
                    MLIRBuilder.line builder (sprintf "llvm.store %s, %s : %s, !llvm.ptr" zero ptr varType)

                // Register the pointer for later store operations
                SSAContext.registerLocal ctx (name + "_ptr") ptr
                // Also register current value for reads - FIDELITY: preserve type
                let loadedVal = SSAContext.nextValue ctx
                MLIRBuilder.line builder (sprintf "%s = llvm.load %s : !llvm.ptr -> %s" loadedVal ptr varType)
                SSAContext.registerLocalWithType ctx name loadedVal varType
                MLIRBuilder.line builder (sprintf "// Mutable var %s: ptr=%s, value=%s, type=%s" name ptr loadedVal varType)
                None
            | None ->
                MLIRBuilder.line builder "// Mutable binding with no name"
                None
        else
            // Emit the expression (immutable binding)
            match exprNode with
            | Some expr ->
                let result = emitExpression builder ctx psg expr
                // Register the SSA name if we have a variable name - FIDELITY: preserve type
                match varName, result with
                | Some name, Some ssaName ->
                    // FIDELITY: Check SSA type first (from emitted code), fall back to PSG type
                    let actualType =
                        match SSAContext.lookupSSAType ctx ssaName with
                        | Some ssaType -> ssaType  // Use type from emitted expression (most accurate)
                        | None -> varType  // Fall back to PSG-derived type
                    MLIRBuilder.line builder (sprintf "// Registering local: %s = %s (type: %s)" name ssaName actualType)
                    SSAContext.registerLocalWithType ctx name ssaName actualType
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

    | sk when sk.StartsWith("MutableSet:") ->
        // Mutable variable assignment: counter <- counter + 1
        let varName = sk.Substring("MutableSet:".Length)
        let children = getChildren psg node

        // Emit the RHS expression
        match children |> List.tryLast with
        | Some rhsNode ->
            match emitExpression builder ctx psg rhsNode with
            | Some rhsValue ->
                // Look up the stack slot for this mutable variable - FIDELITY: use correct type
                match SSAContext.lookupLocal ctx (varName + "_ptr") with
                | Some ptrName ->
                    let varType = SSAContext.lookupLocalType ctx varName |> Option.defaultValue "i32"
                    MLIRBuilder.line builder (sprintf "llvm.store %s, %s : %s, !llvm.ptr" rhsValue ptrName varType)
                    // Update the SSA name to reflect the new value
                    SSAContext.registerLocal ctx varName rhsValue
                    None  // Assignment returns unit
                | None ->
                    MLIRBuilder.line builder (sprintf "// MutableSet: no ptr found for %s" varName)
                    SSAContext.registerLocal ctx varName rhsValue
                    None
            | None ->
                MLIRBuilder.line builder (sprintf "// MutableSet: no RHS value for %s" varName)
                None
        | None ->
            MLIRBuilder.line builder (sprintf "// MutableSet: no children for %s" varName)
            None

    | "WhileLoop" ->
        // While loop: emit using cf dialect with basic blocks
        let children = getChildren psg node

        // First child is condition, rest are body statements
        let condNode = children |> List.tryHead

        // Generate unique block labels
        let loopId = ctx.Counter
        ctx.Counter <- ctx.Counter + 1
        let condBlock = sprintf "while_cond_%d" loopId
        let bodyBlock = sprintf "while_body_%d" loopId
        let exitBlock = sprintf "while_exit_%d" loopId

        // Branch to condition block
        MLIRBuilder.line builder (sprintf "cf.br ^%s" condBlock)

        // Condition block
        MLIRBuilder.lineNoIndent builder (sprintf "^%s:" condBlock)

        // Emit condition
        match condNode with
        | Some cond ->
            match emitExpression builder ctx psg cond with
            | Some condResult ->
                // Convert to i1 if needed (comparison should already be i1)
                MLIRBuilder.line builder (sprintf "cf.cond_br %s, ^%s, ^%s" condResult bodyBlock exitBlock)
            | None ->
                // Default: always enter loop once then exit (shouldn't happen)
                let trueVal = SSAContext.nextValue ctx
                MLIRBuilder.line builder (sprintf "%s = arith.constant true" trueVal)
                MLIRBuilder.line builder (sprintf "cf.cond_br %s, ^%s, ^%s" trueVal bodyBlock exitBlock)
        | None ->
            let trueVal = SSAContext.nextValue ctx
            MLIRBuilder.line builder (sprintf "%s = arith.constant true" trueVal)
            MLIRBuilder.line builder (sprintf "cf.cond_br %s, ^%s, ^%s" trueVal bodyBlock exitBlock)

        // Body block
        MLIRBuilder.lineNoIndent builder (sprintf "^%s:" bodyBlock)
        // All children after the first (condition) are body statements
        let bodyNodes = children |> List.skip 1
        for body in bodyNodes do
            emitExpression builder ctx psg body |> ignore

        // Branch back to condition
        MLIRBuilder.line builder (sprintf "cf.br ^%s" condBlock)

        // Exit block
        MLIRBuilder.lineNoIndent builder (sprintf "^%s:" exitBlock)
        None  // While loop returns unit

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

        // For freestanding entry points, emit exit syscall before return
        // This ensures clean program termination without libc
        if isEntryPoint then
            let exitCode =
                match lastResult with
                | Some result -> result
                | None ->
                    let zero = SSAContext.nextValue ctx
                    MLIRBuilder.line builder (sprintf "%s = arith.constant 0 : i32" zero)
                    zero
            // Extend exit code to i64 for syscall
            let exitCode64 = SSAContext.nextValue ctx
            MLIRBuilder.line builder (sprintf "%s = arith.extsi %s : i32 to i64" exitCode64 exitCode)
            // Emit exit syscall (syscall 60 on Linux x86-64)
            let syscallNum = SSAContext.nextValue ctx
            MLIRBuilder.line builder (sprintf "%s = arith.constant 60 : i64" syscallNum)
            let result = SSAContext.nextValue ctx
            MLIRBuilder.line builder (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"=r,{rax},{rdi}\" %s, %s : (i64, i64) -> i64" result syscallNum exitCode64)

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
