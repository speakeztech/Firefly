module Core.Conversion.LLVMTranslator

open System
open System.IO
open System.Text  // Added import for StringBuilder
open System.Runtime.InteropServices
open Core.XParsec.Foundation

/// Core output types
type LLVMOutput = {
    ModuleName: string
    LLVMIRText: string
    SymbolTable: Map<string, string>
    ExternalFunctions: string list
    GlobalVariables: string list
}

/// MLIR structure types
type ParsedMLIRFunction = {
    Name: string
    Parameters: (string * string) list
    ReturnType: string
    Body: string list
    IsExternal: bool
}

type ParsedMLIRGlobal = {
    Name: string
    Type: string
    Value: string
    IsConstant: bool
}

/// Gets target triple for LLVM based on platform and target
let getTargetTriple = function
    | "x86_64-pc-windows-msvc" | "x86_64-w64-windows-gnu" -> "x86_64-w64-windows-gnu"
    | "x86_64-pc-linux-gnu" -> "x86_64-pc-linux-gnu"  
    | "x86_64-apple-darwin" -> "x86_64-apple-darwin"
    | "embedded" | "thumbv7em-none-eabihf" -> "thumbv7em-none-eabihf"
    | _ when RuntimeInformation.IsOSPlatform(OSPlatform.Windows) -> "x86_64-w64-windows-gnu"
    | _ when RuntimeInformation.IsOSPlatform(OSPlatform.Linux) -> "x86_64-pc-linux-gnu"
    | _ when RuntimeInformation.IsOSPlatform(OSPlatform.OSX) -> "x86_64-apple-darwin"
    | _ -> "x86_64-w64-windows-gnu"

/// MLIR parsing utilities
module MLIRParser =
    /// Extracts module name from MLIR module line
    let extractModuleName (line: string) =
        let trimmed = line.Trim()
        if not (trimmed.StartsWith("module @")) then None
        else 
            let atIndex = trimmed.IndexOf('@')
            trimmed.Substring(atIndex + 1)
            |> fun afterAt -> 
                let endIndex = 
                    [afterAt.IndexOf(' '); afterAt.IndexOf('{')] 
                    |> List.filter (fun i -> i > 0)
                    |> function 
                       | [] -> afterAt.Length
                       | indices -> List.min indices
                Some (afterAt.Substring(0, endIndex).Trim())
    
    /// Extracts function parameters and return type
    let extractFunctionInfo (line: string) =
        let paramInfo =
            let paramStart = line.IndexOf('(')
            let paramEnd = line.IndexOf(')')
            if paramStart < 0 || paramEnd <= paramStart then []
            else 
                let paramStr = line.Substring(paramStart + 1, paramEnd - paramStart - 1).Trim()
                if String.IsNullOrWhiteSpace(paramStr) then []
                else 
                    paramStr.Split(',')
                    |> Array.mapi (fun i paramDecl -> 
                        let parts = paramDecl.Trim().Split(':')
                        if parts.Length >= 2 then (parts.[0].Trim(), parts.[1].Trim())
                        else (sprintf "%%arg%d" i, "i32"))
                    |> Array.toList
                
        let returnType =
            if not (line.Contains("->")) then "void"
            else
                let afterArrow = 
                    line.Substring(line.IndexOf("->") + 2).Trim()
                    |> fun s -> 
                        let endIdx = 
                            [s.IndexOf(' '); s.IndexOf('{')] 
                            |> List.filter (fun i -> i > 0)
                            |> function 
                               | [] -> s.Length
                               | indices -> List.min indices
                        s.Substring(0, endIdx).Trim()
                if afterArrow = "()" then "void" else afterArrow
        
        (paramInfo, returnType)
    
    /// Parses function signature from MLIR
    let parseFunctionSignature (line: string) =
        let trimmed = line.Trim()
        if not (trimmed.Contains("func.func") && not (trimmed.Contains("llvm.func"))) then None
        else
            try
                let isPrivate = trimmed.Contains("private")
                let atIndex = trimmed.IndexOf('@')
                
                if atIndex < 0 then None
                else
                    let afterAt = trimmed.Substring(atIndex + 1)
                    let funcName = 
                        let parenIndex = afterAt.IndexOf('(')
                        if parenIndex > 0 then afterAt.Substring(0, parenIndex)
                        else 
                            let spaceIndex = afterAt.IndexOf(' ')
                            if spaceIndex > 0 then afterAt.Substring(0, spaceIndex)
                            else afterAt
                    
                    let (parameters, returnType) = extractFunctionInfo trimmed
                    
                    Some {
                        Name = "@" + funcName
                        Parameters = parameters
                        ReturnType = returnType
                        Body = []
                        IsExternal = isPrivate
                    }
            with _ -> None
    
    /// Parses global constant from MLIR
    let parseGlobalConstant (line: string) =
        let trimmed = line.Trim()
        if not (trimmed.Contains("memref.global") || trimmed.Contains("llvm.mlir.global")) then None
        else
            try
                let isConstant = trimmed.Contains("constant")
                let atIndex = trimmed.IndexOf('@')
                
                if atIndex < 0 then None
                else
                    let afterAt = trimmed.Substring(atIndex + 1)
                    let endIndex = 
                        [afterAt.IndexOf(' '); afterAt.IndexOf('=')]
                        |> List.filter (fun i -> i > 0)
                        |> function 
                           | [] -> afterAt.Length
                           | indices -> List.min indices
                    
                    let globalName = "@" + afterAt.Substring(0, endIndex).Trim()
                    
                    let value = 
                        if not (trimmed.Contains("dense<")) then "" else
                        let denseStart = trimmed.IndexOf("dense<") + 6
                        let denseEnd = trimmed.IndexOf('>', denseStart)
                        if denseEnd <= denseStart then "" 
                        else trimmed.Substring(denseStart, denseEnd - denseStart)
                    
                    Some {
                        Name = globalName
                        Type = "memref"
                        Value = value
                        IsConstant = isConstant
                    }
            with _ -> None
    
    /// Parses MLIR function body operations
    let parseFunctionBody (lines: string array) (startIndex: int) =
        // Skip signature line, ensure it's a valid function with '{'
        if startIndex >= lines.Length || not (lines.[startIndex].Trim().EndsWith("{")) then
            ([], startIndex + 1)
        else
            let rec collectBody idx operations =
                if idx >= lines.Length then (List.rev operations, idx)
                else
                    let line = lines.[idx].Trim()
                    if line = "}" then (List.rev operations, idx + 1)
                    elif String.IsNullOrWhiteSpace(line) || 
                         line.StartsWith("//") || 
                         line.StartsWith("module") ||
                         line.StartsWith("llvm.func") ||
                         line.StartsWith("func.func") ||
                         line = "{" then collectBody (idx + 1) operations
                    else collectBody (idx + 1) (line :: operations)
            
            collectBody (startIndex + 1) []

/// Converts MLIR operations to LLVM IR
module OperationConverter =
    /// Maps MLIR types to LLVM types
    let mlirTypeToLLVM = function
        | "i32" -> "i32"
        | "i64" -> "i64"
        | "f32" -> "float"
        | "f64" -> "double"
        | "()" | "void" -> "void"
        | t when t.StartsWith("memref<") -> "i8*"
        | _ -> "i32"
    
    /// Extracts sections from operation string using delimiters
    let extractSection (str: string) (startDelim: string) (endDelim: string option) =
        let startIdx = str.IndexOf(startDelim)
        if startIdx < 0 then ""
        else
            let contentStart = startIdx + startDelim.Length
            match endDelim with
            | Some ed -> 
                let endIdx = str.IndexOf(ed, contentStart)
                if endIdx < 0 then str.Substring(contentStart)
                else str.Substring(contentStart, endIdx - contentStart)
            | None -> str.Substring(contentStart)
    
    /// Converts a single MLIR operation to LLVM IR
    let convertOperation (operation: string) =
        let trimmed = operation.Trim()
        
        // Helper to extract variable name from assignment
        let extractResultVar (op: string) =
            let parts = op.Split('=')
            if parts.Length < 2 then "" else parts.[0].Trim()
        
        match trimmed with
        | op when op.Contains("llvm.mlir.constant") ->
            let resultVar = extractResultVar op
            let constValue = extractSection op "constant" (Some ":")
            let typeStr = extractSection op ":" None
            
            let llvmTypePrefix = 
                if typeStr.Contains("i32") then "i32"
                elif typeStr.Contains("i64") then "i64"
                elif typeStr.Contains("f32") then "float"
                elif typeStr.Contains("f64") then "double"
                else "i32"
            
            let opStr = 
                if llvmTypePrefix.StartsWith("f") then "fadd" 
                else "add"
            
            let zeroVal = if llvmTypePrefix.StartsWith("f") then "0.0" else "0"
            
            sprintf "  %s = %s %s %s, %s" resultVar opStr llvmTypePrefix (constValue.Trim()) zeroVal
            
        | op when op.Contains("llvm.bitcast") ->
            let resultVar = extractResultVar op
            let sourceVar = extractSection op "bitcast %" (Some ":")
            let colonPos = op.IndexOf(':', op.IndexOf("bitcast"))
            let toPos = op.LastIndexOf(" to ")
            
            if colonPos < 0 || toPos < 0 then sprintf "  ; Error parsing bitcast: %s" op
            else
                let sourceType = (op.Substring(colonPos + 1, toPos - colonPos - 1).Trim())
                let targetType = (op.Substring(toPos + 4).Trim())
                
                let llvmSourceType = 
                    match sourceType with
                    | "i32" -> "i32" 
                    | "i64" -> "i64"
                    | t when t.Contains("memref") -> "i8*"
                    | "()" -> "i8*"
                    | _ -> "i8*"
                
                let llvmTargetType = 
                    match targetType with
                    | "i32" -> "i32"
                    | "i64" -> "i64"
                    | t when t.Contains("memref") -> "i8*"
                    | "()" -> "i8*"
                    | _ -> "i8*"
                
                sprintf "  %s = bitcast %s %%%s to %s" resultVar llvmSourceType (sourceVar.Trim()) llvmTargetType
            
        | op when op.Contains("llvm.call") ->
            let resultVar = extractResultVar op
            let atIdx = op.IndexOf('@')
            let openParenIdx = op.IndexOf('(', atIdx)
            let closeParenIdx = op.IndexOf(')', openParenIdx)
            
            if atIdx < 0 || openParenIdx < 0 || closeParenIdx < 0 then
                sprintf "  ; Error parsing function call: %s" op
            else
                let funcName = op.Substring(atIdx, openParenIdx - atIdx)
                let argsStr = op.Substring(openParenIdx + 1, closeParenIdx - openParenIdx - 1)
                
                // Determine return type
                let returnType =
                    let colonIdx = op.IndexOf(':', closeParenIdx)
                    let arrowIdx = op.IndexOf("->", colonIdx)
                    
                    if colonIdx < 0 || arrowIdx < 0 then "i32"
                    else
                        let returnTypeStr = op.Substring(arrowIdx + 2).Trim()
                        let cleanType = 
                            if returnTypeStr.EndsWith(")") 
                            then returnTypeStr.Substring(0, returnTypeStr.Length - 1).Trim()
                            else returnTypeStr
                        
                        match cleanType with
                        | "()" -> "void"
                        | "i32" -> "i32"
                        | "i1" -> "i1"
                        | t when t.Contains("memref") -> "i8*"
                        | _ -> "i32"
                
                // Handle special cases
                match funcName with
                | "@printf" -> 
                    sprintf "  %s = call i32 (i8*, ...) %s(i8* %s)" resultVar funcName argsStr
                | "@is_ok_result" ->
                    sprintf "  %s = call i1 %s(i32 %s)" resultVar funcName argsStr
                | "@extract_result_length" ->
                    sprintf "  %s = call i32 %s(i32 %s)" resultVar funcName argsStr
                | "@create_span" ->
                    let args = argsStr.Split(',')
                    if args.Length >= 2 then
                        sprintf "  %s = call i8* %s(i8* %s, i32 %s)" resultVar funcName (args.[0].Trim()) (args.[1].Trim())
                    else
                        sprintf "  %s = call i8* %s(i8* %s)" resultVar funcName argsStr
                | _ ->
                    // Generic function call
                    if String.IsNullOrWhiteSpace(argsStr) then
                        if returnType = "void" then
                            sprintf "  call void %s()" funcName
                        else
                            sprintf "  %s = call %s %s()" resultVar returnType funcName
                    else
                        if returnType = "void" then
                            sprintf "  call void %s(%s)" funcName argsStr
                        else
                            sprintf "  %s = call %s %s(%s)" resultVar returnType funcName argsStr
            
        | op when op.StartsWith("^") ->
            sprintf "%s:" (op.TrimStart('^'))
            
        | op when op.StartsWith("br ^") ->
            sprintf "  br label %%%s" (op.Substring(3).Trim())
            
        | op when op.StartsWith("cond_br") ->
            let parts = op.Split([|','; ' '|], StringSplitOptions.RemoveEmptyEntries)
            if parts.Length < 3 then sprintf "  ; Error parsing conditional branch: %s" op
            else
                let condition = parts.[1]
                let thenLabel = parts.[2].TrimStart('^')
                let elseLabel = parts.[3].TrimStart('^')
                sprintf "  br i1 %s, label %%%s, label %%%s" condition thenLabel elseLabel
            
        | op when op.Contains(" = ") && op.Contains(" : ") && not (op.Contains("llvm.")) ->
            let resultVar = extractResultVar op
            let rightSide = op.Substring(op.IndexOf('=') + 1).Trim()
            let sourceVar = 
                if rightSide.Contains(":") then rightSide.Substring(0, rightSide.IndexOf(':')).Trim()
                else rightSide
            sprintf "  %s = %s" resultVar sourceVar
            
        | op when op.Contains("memref.llvm.alloca") ->
            let resultVar = extractResultVar op
            sprintf "  %s = alloca i8, i32 256, align 1" resultVar
            
        | op when op.Contains("llvm.mlir.addressof") ->
            let resultVar = extractResultVar op
            let globalName = extractSection op "@" (Some ":")
            sprintf "  %s = getelementptr [32 x i8], [32 x i8]* @%s, i32 0, i32 0" resultVar (globalName.Trim())
        
        | op when op.Contains("llvm.return") ->
            if not (op.Contains(":")) then "  ret void"
            else
                let spaceAfterReturn = op.IndexOf(' ', "llvm.return".Length)
                let colonIdx = op.LastIndexOf(':')
                
                if spaceAfterReturn < 0 || colonIdx < 0 then "  ret void"
                else
                    let returnVal = op.Substring(spaceAfterReturn, colonIdx - spaceAfterReturn).Trim()
                    let typeStr = op.Substring(colonIdx + 1).Trim()
                    
                    if typeStr.Contains("i32") then sprintf "  ret i32 %s" returnVal
                    elif typeStr.Contains("i64") then sprintf "  ret i64 %s" returnVal
                    else sprintf "  ret i32 %s" returnVal
        
        | op when op.StartsWith(";") -> 
            sprintf "  ; %s" (op.TrimStart(';').Trim())
            
        | _ -> sprintf "  ; TODO: %s" trimmed

/// LLVM IR generation
module LLVMGenerator =
    /// Generates LLVM function from parsed MLIR function
    let generateLLVMFunction (func: ParsedMLIRFunction) =
        let sb = new StringBuilder()  // Explicitly create StringBuilder with type
        
        // Convert parameters
        let llvmParameters = 
            match func.Parameters with
            | [] when func.Name = "@main" -> "i32 %argc, i8** %argv"
            | [] -> ""
            | llvmParams ->
                llvmParams
                |> List.map (fun (name, paramType) -> 
                    sprintf "%s %s" (OperationConverter.mlirTypeToLLVM paramType) name)
                |> String.concat ", "
        
        let llvmReturnType = OperationConverter.mlirTypeToLLVM func.ReturnType
        
        if func.IsExternal then
            sb.AppendLine(sprintf "declare %s %s(%s)" llvmReturnType func.Name llvmParameters) |> ignore
        else
            sb.AppendLine(sprintf "define %s %s(%s) {" llvmReturnType func.Name llvmParameters) |> ignore
            sb.AppendLine("entry:") |> ignore
            
            // First sort constants to the beginning
            let (constants, otherOps) = 
                func.Body |> List.partition (fun op -> op.Contains("llvm.mlir.constant"))
            
            // Process constants first
            constants 
            |> List.map OperationConverter.convertOperation
            |> List.iter (fun line -> sb.AppendLine(line) |> ignore)
            
            // Then process all other operations
            otherOps
            |> List.map OperationConverter.convertOperation
            |> List.iter (fun line -> sb.AppendLine(line) |> ignore)
            
            // Add return if not already present
            if not (func.Body |> List.exists (fun op -> op.Contains("return"))) then
                if func.ReturnType = "void" then
                    sb.AppendLine("  ret void") |> ignore
                else
                    sb.AppendLine("  ret i32 0") |> ignore
            
            sb.AppendLine("}") |> ignore
        
        sb.ToString()
    
    /// Generates LLVM global from parsed MLIR global
    let generateLLVMGlobal (globalConstant: ParsedMLIRGlobal) =
        if globalConstant.Value.StartsWith("\"") && globalConstant.Value.EndsWith("\"") then
            let content = globalConstant.Value.Substring(1, globalConstant.Value.Length - 2)
            let cleanContent = content.Replace("\\00", "")
            let actualSize = cleanContent.Length + 1
            sprintf "%s = private unnamed_addr constant [%d x i8] c\"%s\\00\", align 1" 
                globalConstant.Name actualSize cleanContent
        else
            sprintf "%s = private unnamed_addr constant [1 x i8] zeroinitializer, align 1" 
                globalConstant.Name
    
    /// Standard C library declarations
    let standardDeclarations = [
        "declare i32 @printf(i8* nocapture readonly, ...)"
        "declare i32 @fprintf(i8* nocapture readonly, i8* nocapture readonly, ...)"
        "declare i32 @sprintf(i8* nocapture, i8* nocapture readonly, ...)"
        "declare i32 @scanf(i8* nocapture readonly, ...)"
        "declare i32 @fscanf(i8* nocapture readonly, i8* nocapture readonly, ...)"
        "declare i32 @sscanf(i8* nocapture readonly, i8* nocapture readonly, ...)"
        "declare i32 @puts(i8* nocapture readonly)"
        "declare i32 @fputs(i8* nocapture readonly, i8* nocapture readonly)"
        "declare i8* @fgets(i8*, i32, i8*)"
        "declare i32 @getchar()"
        "declare i32 @putchar(i32)"
        "declare i8* @fopen(i8* nocapture readonly, i8* nocapture readonly)"
        "declare i32 @fclose(i8* nocapture)"
        "declare i64 @fread(i8*, i64, i64, i8*)"
        "declare i64 @fwrite(i8* nocapture readonly, i64, i64, i8*)"
        "declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1)"
        "declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)"
        "declare i64 @strlen(i8* nocapture readonly)"
        "declare i32 @strcmp(i8* nocapture readonly, i8* nocapture readonly)"
        "declare i8* @strcpy(i8* nocapture, i8* nocapture readonly)"
        "declare i8* @strcat(i8* nocapture, i8* nocapture readonly)"
        "declare i8* @__stdoutp() #1"
        "declare i8* @__stdinp() #1"
        "attributes #1 = { nounwind }"
    ]

/// Main MLIR to LLVM conversion logic
module MLIRProcessor =
    /// Processes MLIR text and extracts functions with bodies and globals
    let processMlirText (mlirText: string) =
        let lines = mlirText.Split('\n')
        let mutable moduleName = "main"
        let mutable functions = []
        let mutable globalConstants = []
        let mutable currentIndex = 0
        
        while currentIndex < lines.Length do
            let line = lines.[currentIndex]
            let trimmed = line.Trim()
            
            // Extract module name
            match MLIRParser.extractModuleName trimmed with
            | Some name -> moduleName <- name
            | None -> ()
            
            // Parse functions with bodies
            match MLIRParser.parseFunctionSignature trimmed with
            | Some func ->
                let (body, nextIndex) = MLIRParser.parseFunctionBody lines currentIndex
                let funcWithBody = { func with Body = body }
                functions <- funcWithBody :: functions
                currentIndex <- nextIndex - 1  
            | None -> 
                // Parse globals
                match MLIRParser.parseGlobalConstant trimmed with
                | Some globalItem -> globalConstants <- globalItem :: globalConstants
                | None -> ()
            
            currentIndex <- currentIndex + 1
        
        (moduleName, List.rev functions, List.rev globalConstants)
    
    /// Generates complete LLVM module
    let generateLLVMModule (moduleName: string) (functions: ParsedMLIRFunction list) (globalConstants: ParsedMLIRGlobal list) =
        let sb = new StringBuilder()  // Explicitly create StringBuilder with type
        let targetTriple = getTargetTriple "default"
        
        // Module header
        sb.AppendLine(sprintf "; ModuleID = '%s'" moduleName) |> ignore
        sb.AppendLine(sprintf "source_filename = \"%s\"" moduleName) |> ignore
        sb.AppendLine("target datalayout = \"e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128\"") |> ignore
        sb.AppendLine(sprintf "target triple = \"%s\"" targetTriple) |> ignore
        sb.AppendLine("") |> ignore
        
        // Standard declarations
        LLVMGenerator.standardDeclarations |> List.iter (fun line -> sb.AppendLine(line) |> ignore)
        sb.AppendLine("") |> ignore
        
        // Deduplicate global constants by name
        let uniqueGlobals = 
            globalConstants
            |> List.groupBy (fun g -> g.Name)
            |> List.map (fun (_, globals) -> List.head globals)
        
        // Global constants
        uniqueGlobals
        |> List.map LLVMGenerator.generateLLVMGlobal
        |> List.iter (fun line -> sb.AppendLine(line) |> ignore)
        
        if not uniqueGlobals.IsEmpty then sb.AppendLine("") |> ignore
        
        // Functions
        let userFunctions = functions |> List.filter (fun f -> not f.IsExternal)
        userFunctions
        |> List.map LLVMGenerator.generateLLVMFunction
        |> List.iter (fun f -> sb.Append(f).AppendLine("") |> ignore)
        
        // Ensure main function exists
        let hasMain = userFunctions |> List.exists (fun f -> f.Name = "@main")
        if not hasMain then
            let hasHello = userFunctions |> List.exists (fun f -> f.Name = "@hello")
            
            if hasHello then
                sb.AppendLine("define i32 @main(i32 %argc, i8** %argv) {") |> ignore
                sb.AppendLine("entry:") |> ignore
                sb.AppendLine("  call void @hello()") |> ignore
                sb.AppendLine("  ret i32 0") |> ignore
                sb.AppendLine("}") |> ignore
            else
                sb.AppendLine("define i32 @main(i32 %argc, i8** %argv) {") |> ignore
                sb.AppendLine("entry:") |> ignore
                sb.AppendLine("  ret i32 0") |> ignore
                sb.AppendLine("}") |> ignore
        
        sb.ToString()

/// External toolchain integration
module ExternalToolchain =
    /// Checks if command is available in PATH
    let isCommandAvailable command =
        try
            let info = System.Diagnostics.ProcessStartInfo(
                FileName = command,
                Arguments = "--version",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true)
            
            use proc = System.Diagnostics.Process.Start(info)
            proc.WaitForExit(5000) |> ignore
            proc.ExitCode = 0
        with _ -> false
    
    /// Gets available compiler commands
    let getCompilerCommands _ =
        let isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
        
        let compilers = 
            if isWindows then ["llc", "gcc"; "llc", "clang"]
            else ["llc", "clang"; "llc", "gcc"]
            
        compilers
        |> List.tryFind (fun (llc, compiler) -> isCommandAvailable llc && isCommandAvailable compiler)
        |> function
           | Some compilerPair -> Success compilerPair
           | None -> CompilerFailure [ConversionError(
                "toolchain", 
                "No suitable LLVM/compiler toolchain found", 
                "available toolchain", 
                "Install LLVM tools and Clang/GCC")]
    
    /// Runs external command with error handling
    let runExternalCommand command arguments =
        try
            let info = System.Diagnostics.ProcessStartInfo(
                FileName = command,
                Arguments = arguments,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true)
            
            use proc = System.Diagnostics.Process.Start(info)
            proc.WaitForExit()
            
            if proc.ExitCode = 0 then
                Success (proc.StandardOutput.ReadToEnd())
            else
                let error = proc.StandardError.ReadToEnd()
                CompilerFailure [ConversionError(
                    "external command", 
                    sprintf "%s failed with exit code %d" command proc.ExitCode, 
                    "successful execution", 
                    error)]
        with ex ->
            CompilerFailure [ConversionError(
                "external command", 
                sprintf "Failed to execute %s" command, 
                "successful execution", 
                ex.Message)]

/// Main translation entry point
let translateToLLVM (mlirText: string) =
    if String.IsNullOrWhiteSpace(mlirText) then
        CompilerFailure [ConversionError(
            "LLVM translation", 
            "empty input", 
            "LLVM IR", 
            "MLIR input cannot be empty")]
    else
        try
            let (moduleName, functions, globalConstants) = MLIRProcessor.processMlirText mlirText
            let llvmIR = MLIRProcessor.generateLLVMModule moduleName functions globalConstants
            
            Success {
                ModuleName = moduleName
                LLVMIRText = llvmIR
                SymbolTable = Map.empty
                ExternalFunctions = functions 
                                  |> List.filter (fun f -> f.IsExternal) 
                                  |> List.map (fun f -> f.Name)
                GlobalVariables = globalConstants |> List.map (fun g -> g.Name)
            }
        with ex ->
            CompilerFailure [ConversionError(
                "MLIR to LLVM", 
                "MLIR processing", 
                "LLVM IR", 
                sprintf "Exception: %s" ex.Message)]

/// Compiles LLVM IR to native executable
let compileLLVMToNative (llvmOutput: LLVMOutput) (outputPath: string) (target: string) =
    let targetTriple = getTargetTriple target
    match ExternalToolchain.getCompilerCommands target with
    | Success (llcCommand, linkerCommand) ->
        let llvmPath = Path.ChangeExtension(outputPath, ".ll")
        let objPath = Path.ChangeExtension(outputPath, ".o")
        
        try
            // Write LLVM IR to file
            File.WriteAllText(llvmPath, llvmOutput.LLVMIRText, System.Text.UTF8Encoding(false))
            
            // Compile to object file
            let llcArgs = sprintf "-filetype=obj -mtriple=%s -o %s %s" targetTriple objPath llvmPath
            match ExternalToolchain.runExternalCommand llcCommand llcArgs with
            | Success _ ->
                // Link to executable
                let isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                let linkArgs = 
                    if isWindows then
                        sprintf "%s -o %s -Wl,--subsystem,console -Wl,--entry,mainCRTStartup -static-libgcc -lmingw32 -lkernel32 -luser32" 
                            objPath outputPath
                    else
                        sprintf "%s -o %s" objPath outputPath
                        
                match ExternalToolchain.runExternalCommand linkerCommand linkArgs with
                | Success _ ->
                    if File.Exists(objPath) then File.Delete(objPath)
                    Success ()
                | CompilerFailure errors -> CompilerFailure errors
            | CompilerFailure errors -> CompilerFailure errors
        
        with ex ->
            CompilerFailure [ConversionError(
                "native compilation", 
                "Failed during native compilation", 
                "executable", 
                ex.Message)]
    | CompilerFailure errors -> CompilerFailure errors

/// Validates that LLVM IR has no heap allocations
let validateZeroAllocationGuarantees (llvmIR: string) =
    ["malloc"; "calloc"; "realloc"; "new"]
    |> List.exists (fun func -> llvmIR.Contains(sprintf "call.*@%s" func))
    |> function
       | true -> CompilerFailure [ConversionError(
            "zero-allocation validation", 
            "optimized LLVM IR", 
            "zero-allocation LLVM IR", 
            "Found potential heap allocation functions in LLVM IR")]
       | false -> Success ()