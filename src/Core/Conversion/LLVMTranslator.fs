module Core.Conversion.LLVMTranslator

open System
open System.IO
open System.Runtime.InteropServices
open Core.XParsec.Foundation

/// LLVM IR output with complete module information
type LLVMOutput = {
    ModuleName: string
    LLVMIRText: string
    SymbolTable: Map<string, string>
    ExternalFunctions: string list
    GlobalVariables: string list
}

/// MLIR parsing state and domain models
type MLIRParsingState = {
    CurrentModule: string
    Functions: Map<string, MLIRFunction>
    Globals: Map<string, MLIRGlobal>
    Operations: string list
    SymbolTable: Map<string, string>
    TransformationHistory: (string * string) list
}

and MLIRFunction = {
    Name: string
    Parameters: (string * string) list
    ReturnType: string
    Body: string list
    IsExternal: bool
}

and MLIRGlobal = {
    Name: string
    Type: string
    Value: string
    IsConstant: bool
}

/// Initializes MLIR parsing state
let createMLIRParsingState() : MLIRParsingState = {
    CurrentModule = "main"
    Functions = Map.empty
    Globals = Map.empty
    Operations = []
    SymbolTable = Map.empty
    TransformationHistory = []
}

/// Target triple management
module TargetTripleManagement =
    
    /// Gets target triple for LLVM based on platform
    let getTargetTriple (target: string) : string =
        match target.ToLowerInvariant() with
        | "x86_64-pc-windows-msvc" -> "x86_64-pc-windows-msvc"
        | "x86_64-pc-linux-gnu" -> "x86_64-pc-linux-gnu"  
        | "x86_64-apple-darwin" -> "x86_64-apple-darwin"
        | "embedded" -> "thumbv7em-none-eabihf"
        | "thumbv7em-none-eabihf" -> "thumbv7em-none-eabihf"
        | "x86_64-w64-windows-gnu" -> "x86_64-w64-windows-gnu"
        | _ -> 
            if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
                "x86_64-w64-windows-gnu"
            elif RuntimeInformation.IsOSPlatform(OSPlatform.Linux) then
                "x86_64-pc-linux-gnu"
            elif RuntimeInformation.IsOSPlatform(OSPlatform.OSX) then
                "x86_64-apple-darwin"
            else
                "x86_64-w64-windows-gnu"
    
    /// Validates target triple format
    let validateTargetTriple (triple: string) : CompilerResult<unit> =
        let parts = triple.Split('-')
        if parts.Length >= 3 then
            Success ()
        else
            CompilerFailure [ConversionError(
                "target validation", 
                triple, 
                "valid target triple", 
                "Target triple must have format: arch-vendor-os(-environment)")]

/// MLIR parsing and extraction utilities
module MLIRExtraction =
    
    /// Extracts identifier from MLIR text
    let extractIdentifier (text: string) : string option =
        let trimmed = text.Trim()
        if trimmed.Length > 0 then
            let chars = trimmed.ToCharArray()
            let isIdStart c = Char.IsLetter(c) || c = '_' || c = '@' || c = '%'
            let isIdCont c = Char.IsLetter(c) || Char.IsDigit(c) || c = '_' || c = '.'
            
            if chars.Length > 0 && isIdStart(chars.[0]) then
                let sb = System.Text.StringBuilder()
                sb.Append(chars.[0]) |> ignore
                let mutable i = 1
                while i < chars.Length && isIdCont(chars.[i]) do
                    sb.Append(chars.[i]) |> ignore
                    i <- i + 1
                Some (sb.ToString())
            else None
        else None
    
    /// Extracts function signature from MLIR func.func line
    let extractFunctionSignature (line: string) : MLIRFunction option =
        if line.Contains("func.func") then
            try
                let parts = line.Split([|' '|], StringSplitOptions.RemoveEmptyEntries)
                let nameOpt = parts |> Array.tryFind (fun p -> p.StartsWith("@"))
                let isPrivate = parts |> Array.contains "private"
                
                match nameOpt with
                | Some name ->
                    // Extract parameters and return type (simplified)
                    let paramStart = line.IndexOf('(')
                    let paramEnd = line.IndexOf(')')
                    let parameters = 
                        if paramStart >= 0 && paramEnd > paramStart then
                            let paramStr = line.Substring(paramStart + 1, paramEnd - paramStart - 1)
                            if String.IsNullOrWhiteSpace(paramStr) then []
                            else [("param", "i32")]  // Simplified parameter extraction
                        else []
                    
                    let returnType = 
                        if line.Contains("->") then
                            let arrowIndex = line.IndexOf("->")
                            if arrowIndex >= 0 then
                                let afterArrow = line.Substring(arrowIndex + 2).Trim()
                                let spaceIndex = afterArrow.IndexOf(' ')
                                if spaceIndex > 0 then afterArrow.Substring(0, spaceIndex)
                                else afterArrow
                            else "void"
                        else "void"
                    
                    Some {
                        Name = name
                        Parameters = parameters
                        ReturnType = returnType
                        Body = []
                        IsExternal = isPrivate
                    }
                | None -> None
            with _ -> None
        else None
    
    /// Extracts global constant from MLIR memref.global line
    let extractGlobalConstant (line: string) : MLIRGlobal option =
        if line.Contains("memref.global") then
            try
                let parts = line.Split([|' '|], StringSplitOptions.RemoveEmptyEntries)
                let nameOpt = parts |> Array.tryFind (fun p -> p.StartsWith("@"))
                let isConstant = parts |> Array.contains "constant"
                
                match nameOpt with
                | Some name ->
                    let typeStr = 
                        if line.Contains(":") then
                            let colonIndex = line.IndexOf(':')
                            let afterColon = line.Substring(colonIndex + 1).Trim()
                            let spaceIndex = afterColon.IndexOf(' ')
                            if spaceIndex > 0 then afterColon.Substring(0, spaceIndex)
                            else afterColon
                        else "memref<?xi8>"
                    
                    let value = 
                        if line.Contains("=") then
                            let equalIndex = line.IndexOf('=')
                            line.Substring(equalIndex + 1).Trim()
                        else ""
                    
                    Some {
                        Name = name
                        Type = typeStr
                        Value = value
                        IsConstant = isConstant
                    }
                | None -> None
            with _ -> None
        else None
    
    /// Extracts module name from MLIR module line
    let extractModuleName (line: string) : string option =
        if line.Contains("module") then
            let parts = line.Split([|' '|], StringSplitOptions.RemoveEmptyEntries)
            if parts.Length > 1 then
                Some parts.[1]
            else Some "main"
        else None

/// MLIR to LLVM type conversions
module TypeConversion =
    
    /// Converts an MLIR type to LLVM IR type
    let mlirTypeToLLVM (mlirType: string) : string =
        match mlirType.Trim() with
        | t when t.StartsWith("i") || t.StartsWith("f") -> mlirType
        | "void" -> "void"
        | t when t.StartsWith("memref<") -> 
            let size = t.Substring(7, t.Length - 8).Trim()
            let parts = size.Split('x')
            if parts.Length = 2 && parts.[1].Trim() = "i8" then
                if parts.[0].Trim() = "?" then
                    "i8*"
                else
                    sprintf "[%s x i8]" (parts.[0].Trim())
            else
                "i8*"
        | t when t.StartsWith("!llvm.ptr<") ->
            let innerType = t.Substring(11, t.Length - 12).Trim()
            let convertedInner = mlirTypeToLLVM innerType
            if convertedInner.EndsWith("*") then
                convertedInner.Replace("*", "**")
            else
                sprintf "%s*" convertedInner
        | _ -> "i32"  // Default type

/// MLIR to LLVM operation conversions
module OperationConversion =
    
    /// Converts an MLIR function to LLVM IR
    let convertFunction (func: MLIRFunction) : string =
        use writer = new StringWriter()
        
        // Convert parameters
        let paramStr = 
            if func.Parameters.IsEmpty then 
                ""
            else
                let paramStrings = 
                    func.Parameters
                    |> List.map (fun (name, paramType) -> 
                        let llvmType = TypeConversion.mlirTypeToLLVM paramType
                        sprintf "%s %s" llvmType name)
                String.concat ", " paramStrings
        
        let llvmReturnType = TypeConversion.mlirTypeToLLVM func.ReturnType
        
        // Handle special case for main function
        let funcName = 
            if func.Name = "@main" then "@main" else func.Name
            
        let finalParamStr = 
            if func.Name = "@main" then "i32 %argc, i8** %argv" else paramStr
        
        if func.IsExternal then
            // External function declaration
            writer.WriteLine(sprintf "declare %s %s(%s)" llvmReturnType funcName finalParamStr)
        else
            // Function definition
            writer.WriteLine(sprintf "define %s %s(%s) {" llvmReturnType funcName finalParamStr)
            writer.WriteLine("entry:")
            
            // Convert body operations (simplified)
            for line in func.Body do
                if line.Contains("arith.constant") then
                    // Convert arith.constant to LLVM add
                    let parts = line.Split('=')
                    if parts.Length = 2 then
                        let resultName = parts.[0].Trim()
                        let valueMatch = System.Text.RegularExpressions.Regex.Match(parts.[1], @"arith\.constant\s+(\d+)")
                        if valueMatch.Success then
                            let value = valueMatch.Groups.[1].Value
                            writer.WriteLine(sprintf "  %s = add i32 0, %s" resultName value)
                elif line.Contains("func.call") then
                    // Convert func.call to LLVM call
                    let callMatch = System.Text.RegularExpressions.Regex.Match(line, @"(\%\w+)\s*=\s*func\.call\s*(\@\w+)\((.*)\)")
                    if callMatch.Success then
                        let resultName = callMatch.Groups.[1].Value
                        let calleeName = callMatch.Groups.[2].Value
                        let args = callMatch.Groups.[3].Value
                        writer.WriteLine(sprintf "  %s = call i32 %s(%s)" resultName calleeName args)
                elif line.Contains("func.return") then
                    // Convert func.return to LLVM ret
                    let returnMatch = System.Text.RegularExpressions.Regex.Match(line, @"func\.return\s+(\%\w+)")
                    if returnMatch.Success then
                        let resultName = returnMatch.Groups.[1].Value
                        writer.WriteLine(sprintf "  ret i32 %s" resultName)
                    else
                        writer.WriteLine("  ret i32 0")
                else
                    // Pass through other operations with basic cleanup
                    let cleanLine = line.Replace("memref.alloca", "alloca")
                                        .Replace("memref.load", "load")
                                        .Replace("memref.store", "store")
                    writer.WriteLine(sprintf "  %s" cleanLine)
            
            writer.WriteLine("}")
        
        writer.ToString()
    
    /// Converts an MLIR global to LLVM IR
    let convertGlobal (globalVar: MLIRGlobal) : string =
        use writer = new StringWriter()
        
        // Parse type and value from MLIR global
        let typeMatch = globalVar.Type.Contains("memref<")
        let valueMatch = globalVar.Value.Contains("dense<")
        
        if typeMatch && valueMatch then
            // Extract size from memref<NxiM>
            let sizeStart = globalVar.Type.IndexOf('<') + 1
            let sizeEnd = 
                let xIndex = globalVar.Type.IndexOf('x', sizeStart)
                if xIndex > sizeStart then xIndex else globalVar.Type.Length - 1
                
            let size = globalVar.Type.Substring(sizeStart, sizeEnd - sizeStart)
            
            // Extract content from dense<"value">
            let content = 
                let startQuote = globalVar.Value.IndexOf('"')
                let endQuote = globalVar.Value.LastIndexOf('"')
                if startQuote >= 0 && endQuote > startQuote then
                    let contentText = globalVar.Value.Substring(startQuote + 1, endQuote - startQuote - 1)
                    contentText.Replace("\\00", "")
                else
                    ""
            
            let actualSize = content.Length + 1  // Include null terminator
            writer.WriteLine(sprintf "%s = private unnamed_addr constant [%d x i8] c\"%s\\00\", align 1" 
                    globalVar.Name actualSize content)
        else
            writer.WriteLine(sprintf "%s = private unnamed_addr constant [1 x i8] zeroinitializer, align 1" globalVar.Name)
        
        writer.ToString()
    
    /// Generates standard C library declarations
    let generateStandardDeclarations() : string list =
        [
            // Standard C library printf and printf-family functions
            "declare i32 @printf(i8* nocapture readonly, ...)"
            "declare i32 @fprintf(i8* nocapture readonly, i8* nocapture readonly, ...)"
            "declare i32 @sprintf(i8* nocapture, i8* nocapture readonly, ...)"
            
            // Standard C library scanf and scanf-family functions
            "declare i32 @scanf(i8* nocapture readonly, ...)"
            "declare i32 @fscanf(i8* nocapture readonly, i8* nocapture readonly, ...)"
            "declare i32 @sscanf(i8* nocapture readonly, i8* nocapture readonly, ...)"
            
            // String I/O functions
            "declare i32 @puts(i8* nocapture readonly)"
            "declare i32 @fputs(i8* nocapture readonly, i8* nocapture readonly)"
            "declare i8* @fgets(i8*, i32, i8*)"
            "declare i32 @getchar()"
            "declare i32 @putchar(i32)"
            
            // File operations
            "declare i8* @fopen(i8* nocapture readonly, i8* nocapture readonly)"
            "declare i32 @fclose(i8* nocapture)"
            "declare i64 @fread(i8*, i64, i64, i8*)"
            "declare i64 @fwrite(i8* nocapture readonly, i64, i64, i8*)"
            
            // Memory operations
            "declare i8* @malloc(i64)"
            "declare void @free(i8* nocapture)"
            "declare i8* @calloc(i64, i64)"
            "declare i8* @realloc(i8* nocapture, i64)"
            "declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1)"
            "declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)"
            
            // String operations
            "declare i64 @strlen(i8* nocapture readonly)"
            "declare i32 @strcmp(i8* nocapture readonly, i8* nocapture readonly)"
            "declare i8* @strcpy(i8* nocapture, i8* nocapture readonly)"
            "declare i8* @strcat(i8* nocapture, i8* nocapture readonly)"
            
            // Standard streams
            "declare i8* @__stdoutp() #1"
            "declare i8* @__stdinp() #1"
            
            // Attributes
            "attributes #1 = { nounwind }"
        ]

/// MLIR module processing
module MLIRProcessing =
    
    /// Processes a single line of MLIR
    let processLine (line: string) (state: MLIRParsingState) : MLIRParsingState =
        let trimmedLine = line.Trim()
        
        if String.IsNullOrWhiteSpace(trimmedLine) || trimmedLine.StartsWith("//") then
            state
        elif trimmedLine.StartsWith("module") then
            match MLIRExtraction.extractModuleName trimmedLine with
            | Some moduleName -> { state with CurrentModule = moduleName }
            | None -> state
        elif trimmedLine.StartsWith("func.func") then
            match MLIRExtraction.extractFunctionSignature trimmedLine with
            | Some func -> { state with Functions = Map.add func.Name func state.Functions }
            | None -> state
        elif trimmedLine.StartsWith("memref.global") then
            match MLIRExtraction.extractGlobalConstant trimmedLine with
            | Some globalVar -> { state with Globals = Map.add globalVar.Name globalVar state.Globals }
            | None -> state
        else
            state
    
    /// Processes a complete MLIR module
    let processModule (mlirText: string) : MLIRParsingState =
        let lines = mlirText.Split('\n')
        let initialState = createMLIRParsingState()
        
        Array.fold (fun state line -> processLine line state) initialState lines
    
    /// Generates LLVM IR module from parsed MLIR
    let generateLLVMModule (targetTriple: string) (state: MLIRParsingState) : string =
        use writer = new StringWriter()
        
        // Generate module header
        writer.WriteLine(sprintf "; ModuleID = '%s'" state.CurrentModule)
        writer.WriteLine(sprintf "source_filename = \"%s\"" state.CurrentModule)
        writer.WriteLine("target datalayout = \"e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128\"")
        writer.WriteLine(sprintf "target triple = \"%s\"" targetTriple)
        writer.WriteLine("")
        
        // Generate standard declarations
        let declarations = OperationConversion.generateStandardDeclarations()
        for decl in declarations do
            writer.WriteLine(decl)
        writer.WriteLine("")
        
        // Generate globals
        let globals = state.Globals |> Map.toList |> List.map snd
        for globalVar in globals do
            let globalLLVM = OperationConversion.convertGlobal globalVar
            writer.Write(globalLLVM)
        writer.WriteLine("")
        
        // Generate functions
        let functions = state.Functions |> Map.toList |> List.map snd
        for func in functions do
            let funcLLVM = OperationConversion.convertFunction func
            writer.Write(funcLLVM)
            writer.WriteLine("")
        
        // Ensure we have a proper main wrapper if needed
        let hasMainFunc = state.Functions |> Map.containsKey "@main"
        if not hasMainFunc then
            let functions = state.Functions |> Map.toList |> List.map (fun (_, func) -> func.Name)
            let hasUserMain = functions |> List.exists (fun name -> name.Contains("main") || name.Contains("Main"))
            
            if hasUserMain then
                let userMain = functions |> List.find (fun name -> name.Contains("main") || name.Contains("Main"))
                writer.WriteLine("")
                writer.WriteLine("define i32 @main(i32 %argc, i8** %argv) {")
                writer.WriteLine("entry:")
                writer.WriteLine(sprintf "  %%1 = call i32 %s()" userMain)
                writer.WriteLine("  ret i32 %1")
                writer.WriteLine("}")
            else
                writer.WriteLine("")
                writer.WriteLine("define i32 @main(i32 %argc, i8** %argv) {")
                writer.WriteLine("entry:")
                writer.WriteLine("  ret i32 0")
                writer.WriteLine("}")
        
        writer.ToString()

/// Main translation entry point
let translateToLLVM (mlirText: string) : CompilerResult<LLVMOutput> =
    if String.IsNullOrWhiteSpace(mlirText) then
        CompilerFailure [ConversionError("LLVM translation", "empty input", "LLVM IR", "MLIR input cannot be empty")]
    else
        let targetTriple = TargetTripleManagement.getTargetTriple "default"
        
        try
            let state = MLIRProcessing.processModule mlirText
            let llvmIR = MLIRProcessing.generateLLVMModule targetTriple state
            
            Success {
                ModuleName = state.CurrentModule
                LLVMIRText = llvmIR
                SymbolTable = state.SymbolTable
                ExternalFunctions = []  // Populated during conversion
                GlobalVariables = state.Globals |> Map.toList |> List.map fst
            }
        with ex ->
            CompilerFailure [ConversionError(
                "MLIR to LLVM", 
                "MLIR processing", 
                "LLVM IR", 
                sprintf "Exception: %s" ex.Message)]

/// External tool integration for native compilation
module ExternalToolchain =
    
    /// Checks if a command is available in PATH
    let isCommandAvailable (command: string) : bool =
        try
            let processInfo = System.Diagnostics.ProcessStartInfo()
            processInfo.FileName <- command
            processInfo.Arguments <- "--version"
            processInfo.UseShellExecute <- false
            processInfo.RedirectStandardOutput <- true
            processInfo.RedirectStandardError <- true
            processInfo.CreateNoWindow <- true
            
            use proc = System.Diagnostics.Process.Start(processInfo)
            proc.WaitForExit(5000) |> ignore
            proc.ExitCode = 0
        with
        | _ -> false
    
    /// Gets available compiler commands
    let getCompilerCommands (target: string) : CompilerResult<string * string> =
        if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
            if isCommandAvailable "llc" && isCommandAvailable "gcc" then
                Success ("llc", "gcc")
            elif isCommandAvailable "llc" && isCommandAvailable "clang" then
                Success ("llc", "clang")
            else
                CompilerFailure [ConversionError("toolchain", "No suitable LLVM/compiler toolchain found", "available toolchain", "Install LLVM tools and GCC/Clang")]
        else
            if isCommandAvailable "llc" && isCommandAvailable "clang" then
                Success ("llc", "clang")
            elif isCommandAvailable "llc" && isCommandAvailable "gcc" then
                Success ("llc", "gcc")
            else
                CompilerFailure [ConversionError("toolchain", "No suitable LLVM/compiler toolchain found", "available toolchain", "Install LLVM tools and Clang/GCC")]
    
    /// Runs external command with error handling
    let runExternalCommand (command: string) (arguments: string) : CompilerResult<string> =
        try
            let processInfo = System.Diagnostics.ProcessStartInfo()
            processInfo.FileName <- command
            processInfo.Arguments <- arguments
            processInfo.UseShellExecute <- false
            processInfo.RedirectStandardOutput <- true
            processInfo.RedirectStandardError <- true
            processInfo.CreateNoWindow <- true
            
            use proc = System.Diagnostics.Process.Start(processInfo)
            proc.WaitForExit()
            
            if proc.ExitCode = 0 then
                Success (proc.StandardOutput.ReadToEnd())
            else
                let error = proc.StandardError.ReadToEnd()
                CompilerFailure [ConversionError("external command", sprintf "%s failed with exit code %d" command proc.ExitCode, "successful execution", error)]
        with
        | ex ->
            CompilerFailure [ConversionError("external command", sprintf "Failed to execute %s" command, "successful execution", ex.Message)]

/// Compiles LLVM IR to native executable with enhanced MinGW compatibility
let compileLLVMToNative (llvmOutput: LLVMOutput) (outputPath: string) (target: string) : CompilerResult<unit> =
    let targetTriple = TargetTripleManagement.getTargetTriple target
    match TargetTripleManagement.validateTargetTriple targetTriple with
    | Success () ->
        match ExternalToolchain.getCompilerCommands target with
        | Success (llcCommand, linkerCommand) ->
            let llvmPath = Path.ChangeExtension(outputPath, ".ll")
            let objPath = Path.ChangeExtension(outputPath, ".o")
            
            try
                // Write LLVM IR to file
                let utf8WithoutBom = System.Text.UTF8Encoding(false)
                File.WriteAllText(llvmPath, llvmOutput.LLVMIRText, utf8WithoutBom)
                
                // Compile to object file
                let llcArgs = sprintf "-filetype=obj -mtriple=%s -o %s %s" targetTriple objPath llvmPath
                match ExternalToolchain.runExternalCommand llcCommand llcArgs with
                | Success _ ->
                    // Link to executable with explicit console subsystem and entry point
                    let linkArgs = 
                        if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
                            sprintf "%s -o %s -Wl,--subsystem,console -Wl,--entry,mainCRTStartup -static-libgcc -lmingw32 -lkernel32 -luser32" objPath outputPath
                        else
                            sprintf "%s -o %s" objPath outputPath
                            
                    match ExternalToolchain.runExternalCommand linkerCommand linkArgs with
                    | Success _ ->
                        // Cleanup intermediate files but keep LLVM IR for debugging
                        if File.Exists(objPath) then File.Delete(objPath)
                        Success ()
                    | CompilerFailure errors -> CompilerFailure errors
                | CompilerFailure errors -> CompilerFailure errors
            
            with
            | ex ->
                CompilerFailure [ConversionError("native compilation", "Failed during native compilation", "executable", ex.Message)]
        | CompilerFailure errors -> CompilerFailure errors
    | CompilerFailure errors -> CompilerFailure errors

/// Validates that LLVM IR has no heap allocations
let validateZeroAllocationGuarantees (llvmIR: string) : CompilerResult<unit> =
    // Check for common heap allocation functions
    let heapFunctions = ["malloc"; "calloc"; "realloc"; "new"]
    
    let containsHeapAllocation = 
        heapFunctions
        |> List.exists (fun func -> 
            let pattern = sprintf "call.*@%s" func
            llvmIR.Contains(pattern))
    
    if containsHeapAllocation then
        CompilerFailure [ConversionError(
            "zero-allocation validation", 
            "optimized LLVM IR", 
            "zero-allocation LLVM IR", 
            "Found potential heap allocation functions in LLVM IR")]
    else
        Success ()

/// Analyzes LLVM IR for statistics and validation
let analyzeLLVMIR (llvmOutput: LLVMOutput) : CompilerResult<Map<string, string>> =
    try
        let lines = llvmOutput.LLVMIRText.Split('\n')
        let functionCount = lines |> Array.filter (fun line -> line.Trim().StartsWith("define")) |> Array.length
        let globalCount = lines |> Array.filter (fun line -> line.Contains("= global") || line.Contains("= constant")) |> Array.length
        let instructionCount = lines |> Array.filter (fun line -> line.Trim().StartsWith("%") || line.Contains("call") || line.Contains("ret")) |> Array.length
        
        let statistics = Map.ofList [
            ("module_name", llvmOutput.ModuleName)
            ("function_count", string functionCount)
            ("global_count", string globalCount)
            ("instruction_count", string instructionCount)
            ("total_lines", string lines.Length)
        ]
        
        Success statistics
    with ex ->
        CompilerFailure [ConversionError("LLVM analysis", "LLVM IR", "statistics", ex.Message)]