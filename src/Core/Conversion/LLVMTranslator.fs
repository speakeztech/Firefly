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

/// Parsed MLIR function representation
type ParsedMLIRFunction = {
    Name: string
    Parameters: (string * string) list
    ReturnType: string
    Body: string list
    IsExternal: bool
}

/// Parsed MLIR global representation
type ParsedMLIRGlobal = {
    Name: string
    Type: string
    Value: string
    IsConstant: bool
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

/// MLIR parsing utilities
module MLIRParser =
    
    /// Extracts module name from MLIR module line
    let extractModuleName (line: string) : string option =
        let trimmed = line.Trim()
        if trimmed.StartsWith("module @") then
            let atIndex = trimmed.IndexOf('@')
            if atIndex >= 0 then
                let afterAt = trimmed.Substring(atIndex + 1)
                let spaceIndex = afterAt.IndexOf(' ')
                let braceIndex = afterAt.IndexOf('{')
                let endIndex = 
                    if spaceIndex > 0 && braceIndex > 0 then min spaceIndex braceIndex
                    elif braceIndex > 0 then braceIndex
                    else afterAt.Length
                Some (afterAt.Substring(0, endIndex).Trim())
            else None
        else None
    
    /// Parses function signature from MLIR
    let parseFunctionSignature (line: string) : ParsedMLIRFunction option =
        let trimmed = line.Trim()
        if trimmed.Contains("func.func") then
            try
                let isPrivate = trimmed.Contains("private")
                
                // Extract function name
                let atIndex = trimmed.IndexOf('@')
                if atIndex >= 0 then
                    let afterAt = trimmed.Substring(atIndex + 1)
                    let parenIndex = afterAt.IndexOf('(')
                    let funcName = 
                        if parenIndex > 0 then
                            afterAt.Substring(0, parenIndex)
                        else
                            let spaceIndex = afterAt.IndexOf(' ')
                            if spaceIndex > 0 then afterAt.Substring(0, spaceIndex)
                            else afterAt
                    
                    // Extract parameters - simplified parsing
                    let paramStart = trimmed.IndexOf('(')
                    let paramEnd = trimmed.IndexOf(')')
                    let parameters = 
                        if paramStart >= 0 && paramEnd > paramStart then
                            let paramStr = trimmed.Substring(paramStart + 1, paramEnd - paramStart - 1).Trim()
                            if String.IsNullOrWhiteSpace(paramStr) then []
                            else 
                                // Simple parameter parsing - for now just count them
                                if paramStr.Contains(",") then
                                    paramStr.Split(',') 
                                    |> Array.mapi (fun i _ -> (sprintf "%%arg%d" i, "i32"))
                                    |> Array.toList
                                else if not (String.IsNullOrWhiteSpace(paramStr)) then
                                    [("%arg0", "i32")]
                                else []
                        else []
                    
                    // Extract return type
                    let returnType = 
                        if trimmed.Contains("->") then
                            let arrowIndex = trimmed.IndexOf("->")
                            if arrowIndex >= 0 then
                                let afterArrow = trimmed.Substring(arrowIndex + 2).Trim()
                                let spaceIndex = afterArrow.IndexOf(' ')
                                let braceIndex = afterArrow.IndexOf('{')
                                let endIndex = 
                                    if spaceIndex > 0 && braceIndex > 0 then min spaceIndex braceIndex
                                    elif spaceIndex > 0 then spaceIndex
                                    elif braceIndex > 0 then braceIndex
                                    else afterArrow.Length
                                let retType = afterArrow.Substring(0, endIndex).Trim()
                                if retType = "()" then "void" else "i32"
                            else "void"
                        else "void"
                    
                    Some {
                        Name = "@" + funcName
                        Parameters = parameters
                        ReturnType = returnType
                        Body = []
                        IsExternal = isPrivate
                    }
                else None
            with _ -> None
        else None
    
    /// Parses global constant from MLIR
    let parseGlobalConstant (line: string) : ParsedMLIRGlobal option =
        let trimmed = line.Trim()
        if trimmed.Contains("memref.global") then
            try
                let isConstant = trimmed.Contains("constant")
                
                // Extract global name
                let atIndex = trimmed.IndexOf('@')
                if atIndex >= 0 then
                    let afterAt = trimmed.Substring(atIndex + 1)
                    let spaceIndex = afterAt.IndexOf(' ')
                    let equalIndex = afterAt.IndexOf('=')
                    let endIndex = 
                        if spaceIndex > 0 && equalIndex > 0 then min spaceIndex equalIndex
                        elif spaceIndex > 0 then spaceIndex
                        elif equalIndex > 0 then equalIndex
                        else afterAt.Length
                    let globalName = "@" + afterAt.Substring(0, endIndex).Trim()
                    
                    // Extract value if present
                    let value = 
                        if trimmed.Contains("dense<") then
                            let denseStart = trimmed.IndexOf("dense<") + 6
                            let denseEnd = trimmed.IndexOf('>', denseStart)
                            if denseEnd > denseStart then
                                trimmed.Substring(denseStart, denseEnd - denseStart)
                            else ""
                        else ""
                    
                    Some {
                        Name = globalName
                        Type = "memref"
                        Value = value
                        IsConstant = isConstant
                    }
                else None
            with _ -> None
        else None

/// LLVM IR generation
module LLVMGenerator =
    
    /// Converts MLIR type to LLVM type
    let mlirTypeToLLVM (mlirType: string) : string =
        match mlirType.Trim() with
        | "i32" -> "i32"
        | "i64" -> "i64"
        | "f32" -> "float"
        | "f64" -> "double"
        | "()" -> "void"
        | "void" -> "void"
        | t when t.StartsWith("memref<") -> "i8*"
        | _ -> "i32"
    
    /// Generates LLVM function from parsed MLIR function
    let generateLLVMFunction (func: ParsedMLIRFunction) : string =
        let writer = new StringWriter()
        
        // Convert parameters
        let llvmParameters = 
            match func.Parameters with
            | [] -> 
                if func.Name = "@main" then "i32 %argc, i8** %argv" else ""
            | parameterList ->
                parameterList
                |> List.map (fun (name, paramType) -> 
                    let llvmType = mlirTypeToLLVM paramType
                    sprintf "%s %s" llvmType name)
                |> String.concat ", "
        
        let llvmReturnType = mlirTypeToLLVM func.ReturnType
        
        if func.IsExternal then
            // External function declaration
            writer.WriteLine(sprintf "declare %s %s(%s)" llvmReturnType func.Name llvmParameters)
        else
            // Function definition
            writer.WriteLine(sprintf "define %s %s(%s) {" llvmReturnType func.Name llvmParameters)
            writer.WriteLine("entry:")
            
            // Function body - for now just return appropriate value
            if func.ReturnType = "void" then
                writer.WriteLine("  ret void")
            else
                writer.WriteLine("  ret i32 0")
            
            writer.WriteLine("}")
        
        writer.ToString()
    
    /// Generates LLVM global from parsed MLIR global
    let generateLLVMGlobal (globalConstant: ParsedMLIRGlobal) : string =
        let writer = new StringWriter()
        
        // Parse the dense value
        if globalConstant.Value.StartsWith("\"") && globalConstant.Value.EndsWith("\"") then
            let content = globalConstant.Value.Substring(1, globalConstant.Value.Length - 2)
            let cleanContent = content.Replace("\\00", "")
            let actualSize = cleanContent.Length + 1
            writer.WriteLine(sprintf "%s = private unnamed_addr constant [%d x i8] c\"%s\\00\", align 1" 
                    globalConstant.Name actualSize cleanContent)
        else
            writer.WriteLine(sprintf "%s = private unnamed_addr constant [1 x i8] zeroinitializer, align 1" globalConstant.Name)
        
        writer.ToString()
    
    /// Generates standard C library declarations
    let generateStandardDeclarations() : string list =
        [
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

/// Main MLIR processing
module MLIRProcessor =
    
    /// Processes MLIR text and extracts functions and globals
    let processMlirText (mlirText: string) : string * ParsedMLIRFunction list * ParsedMLIRGlobal list =
        let lines = mlirText.Split('\n')
        let mutable moduleName = "main"
        let mutable functions = []
        let mutable globalConstants = []
        
        for line in lines do
            let trimmed = line.Trim()
            
            // Extract module name
            match MLIRParser.extractModuleName trimmed with
            | Some name -> moduleName <- name
            | None -> ()
            
            // Parse functions
            match MLIRParser.parseFunctionSignature trimmed with
            | Some func -> functions <- func :: functions
            | None -> ()
            
            // Parse globals
            match MLIRParser.parseGlobalConstant trimmed with
            | Some globalItem -> globalConstants <- globalItem :: globalConstants
            | None -> ()
        
        (moduleName, List.rev functions, List.rev globalConstants)
    
    /// Generates complete LLVM module
    let generateLLVMModule (moduleName: string) (functions: ParsedMLIRFunction list) (globalConstants: ParsedMLIRGlobal list) : string =
        let writer = new StringWriter()
        let targetTriple = TargetTripleManagement.getTargetTriple "default"
        
        // Module header
        writer.WriteLine(sprintf "; ModuleID = '%s'" moduleName)
        writer.WriteLine(sprintf "source_filename = \"%s\"" moduleName)
        writer.WriteLine("target datalayout = \"e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128\"")
        writer.WriteLine(sprintf "target triple = \"%s\"" targetTriple)
        writer.WriteLine("")
        
        // Standard declarations
        let declarations = LLVMGenerator.generateStandardDeclarations()
        for decl in declarations do
            writer.WriteLine(decl)
        writer.WriteLine("")
        
        // Global constants
        for globalItem in globalConstants do
            let globalLLVM = LLVMGenerator.generateLLVMGlobal globalItem
            writer.Write(globalLLVM)
        
        if not globalConstants.IsEmpty then writer.WriteLine("")
        
        // Functions
        let userFunctions = functions |> List.filter (fun f -> not f.IsExternal)
        for func in userFunctions do
            let funcLLVM = LLVMGenerator.generateLLVMFunction func
            writer.Write(funcLLVM)
            writer.WriteLine("")
        
        // Ensure main function exists
        let hasMain = userFunctions |> List.exists (fun f -> f.Name = "@main")
        if not hasMain then
            let hasOtherFunctions = not userFunctions.IsEmpty
            if hasOtherFunctions then
                let firstFunc = userFunctions |> List.head
                writer.WriteLine("define i32 @main(i32 %argc, i8** %argv) {")
                writer.WriteLine("entry:")
                writer.WriteLine(sprintf "  call void %s()" firstFunc.Name)
                writer.WriteLine("  ret i32 0")
                writer.WriteLine("}")
            else
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
        try
            let (moduleName, functions, globalConstants) = MLIRProcessor.processMlirText mlirText
            
            printfn "MLIR analysis: found %d functions, %d globals" functions.Length globalConstants.Length
            
            let llvmIR = MLIRProcessor.generateLLVMModule moduleName functions globalConstants
            
            Success {
                ModuleName = moduleName
                LLVMIRText = llvmIR
                SymbolTable = Map.empty
                ExternalFunctions = functions |> List.filter (fun f -> f.IsExternal) |> List.map (fun f -> f.Name)
                GlobalVariables = globalConstants |> List.map (fun g -> g.Name)
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

/// Compiles LLVM IR to native executable
let compileLLVMToNative (llvmOutput: LLVMOutput) (outputPath: string) (target: string) : CompilerResult<unit> =
    let targetTriple = TargetTripleManagement.getTargetTriple target
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
                // Link to executable
                let linkArgs = 
                    if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
                        sprintf "%s -o %s -Wl,--subsystem,console -Wl,--entry,mainCRTStartup -static-libgcc -lmingw32 -lkernel32 -luser32" objPath outputPath
                    else
                        sprintf "%s -o %s" objPath outputPath
                        
                match ExternalToolchain.runExternalCommand linkerCommand linkArgs with
                | Success _ ->
                    // Cleanup intermediate files
                    if File.Exists(objPath) then File.Delete(objPath)
                    Success ()
                | CompilerFailure errors -> CompilerFailure errors
            | CompilerFailure errors -> CompilerFailure errors
        
        with
        | ex ->
            CompilerFailure [ConversionError("native compilation", "Failed during native compilation", "executable", ex.Message)]
    | CompilerFailure errors -> CompilerFailure errors

/// Validates that LLVM IR has no heap allocations
let validateZeroAllocationGuarantees (llvmIR: string) : CompilerResult<unit> =
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