module Core.Conversion.LLVMTranslator

open System
open System.IO
open System.Runtime.InteropServices
open System.Text.RegularExpressions
open XParsec
open Core.XParsec.Foundation
open Core.XParsec.Foundation.Combinators
open Core.XParsec.Foundation.CharParsers  
open Core.XParsec.Foundation.StringParsers
open Core.XParsec.Foundation.ErrorHandling

/// LLVM IR output with complete module information
type LLVMOutput = {
    ModuleName: string
    LLVMIRText: string
    SymbolTable: Map<string, string>
    ExternalFunctions: string list
    GlobalVariables: string list
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
            CompilerFailure [TransformError(
                "target validation", 
                triple, 
                "valid target triple", 
                "Target triple must have format: arch-vendor-os(-environment)")]

/// Enhanced MLIR parsing and conversion
module EnhancedMLIRProcessing =
    
    /// Represents a parsed MLIR function
    type ParsedMLIRFunction = {
        Name: string
        Parameters: (string * string) list  // (name, type)
        ReturnType: string
        Body: string list
        IsExternal: bool
    }
    
    /// Represents a parsed MLIR global constant
    type ParsedMLIRGlobal = {
        Name: string
        Type: string
        Value: string
        IsConstant: bool
    }
    
    /// Parses MLIR function signature
    let parseFunctionSignature (line: string) : ParsedMLIRFunction option =
        // Handle both regular and external function declarations
        let funcPattern = @"func\.func\s+(?:private\s+)?@(\w+)\s*\(([^)]*)\)\s*(?:->\s*(\w+))?"
        let m = Regex.Match(line.Trim(), funcPattern)
        
        if m.Success then
            let name = m.Groups.[1].Value
            let paramStr = m.Groups.[2].Value
            let returnType = if m.Groups.[3].Success then m.Groups.[3].Value else "void"
            let isExternal = line.Contains("private")
            
            let parameters = 
                if String.IsNullOrWhiteSpace(paramStr) then []
                else
                    paramStr.Split(',')
                    |> Array.map (fun p -> 
                        let parts = p.Trim().Split(':')
                        if parts.Length = 2 then
                            (parts.[0].Trim(), parts.[1].Trim())
                        else
                            ("", ""))
                    |> Array.filter (fun (n, t) -> not (String.IsNullOrEmpty(n) || String.IsNullOrEmpty(t)))
                    |> Array.toList
            
            Some {
                Name = name
                Parameters = parameters
                ReturnType = returnType
                Body = []
                IsExternal = isExternal
            }
        else None
    
    /// Extracts function body from MLIR text
    let extractFunctionBody (lines: string array) (startIndex: int) : string list * int =
        let rec collectBody acc braceCount index =
            if index >= lines.Length then (List.rev acc, index)
            else
                let line = lines.[index].Trim()
                let openBraces = line.ToCharArray() |> Array.filter ((=) '{') |> Array.length
                let closeBraces = line.ToCharArray() |> Array.filter ((=) '}') |> Array.length
                let newBraceCount = braceCount + openBraces - closeBraces
                
                if newBraceCount = 0 && braceCount > 0 then
                    (List.rev (line :: acc), index + 1)
                else
                    collectBody (line :: acc) newBraceCount (index + 1)
        
        collectBody [] 0 startIndex
    
    /// Parses complete MLIR module into structured components
    let parseMLIRModule (mlirText: string) : ParsedMLIRFunction list * ParsedMLIRGlobal list =
        let lines = mlirText.Split('\n')
        let mutable functions = []
        let mutable globals = []
        let mutable i = 0
        
        while i < lines.Length do
            let line = lines.[i].Trim()
            
            if line.StartsWith("func.func") then
                match parseFunctionSignature line with
                | Some funcSig ->
                    if funcSig.IsExternal then
                        // External function - no body
                        functions <- funcSig :: functions
                        i <- i + 1
                    else
                        // Regular function - extract body
                        let (body, nextIndex) = extractFunctionBody lines (i + 1)
                        let completeFunc = { funcSig with Body = body }
                        functions <- completeFunc :: functions
                        i <- nextIndex
                | None ->
                    i <- i + 1
            
            elif line.StartsWith("memref.global") then
                // Parse global constant
                let globalPattern = @"memref\.global\s+constant\s+(@\w+)\s*:\s*([^=]+)=\s*(.+)"
                let m = Regex.Match(line, globalPattern)
                if m.Success then
                    let global = {
                        Name = m.Groups.[1].Value
                        Type = m.Groups.[2].Value.Trim()
                        Value = m.Groups.[3].Value.Trim()
                        IsConstant = true
                    }
                    globals <- global :: globals
                i <- i + 1
            
            else
                i <- i + 1
        
        (List.rev functions, List.rev globals)
    
    /// Converts MLIR instruction to LLVM IR
    let convertMLIRInstruction (instruction: string) : string option =
        let trimmed = instruction.Trim()
        
        if String.IsNullOrWhiteSpace(trimmed) || trimmed.StartsWith("//") then
            None
        elif trimmed.StartsWith("arith.constant") then
            let constPattern = @"(%\w+)\s*=\s*arith\.constant\s+([^:]+):\s*(.+)"
            let m = Regex.Match(trimmed, constPattern)
            if m.Success then
                Some (sprintf "  %s = add %s 0, %s" m.Groups.[1].Value m.Groups.[3].Value m.Groups.[2].Value)
            else None
        
        elif trimmed.StartsWith("func.call") then
            let callPattern = @"(%\w+)\s*=\s*func\.call\s*@(\w+)\(([^)]*)\)\s*:\s*\([^)]*\)\s*->\s*(.+)"
            let m = Regex.Match(trimmed, callPattern)
            if m.Success then
                let result = m.Groups.[1].Value
                let funcName = m.Groups.[2].Value
                let args = m.Groups.[3].Value
                let returnType = m.Groups.[4].Value
                
                // Special handling for I/O functions
                match funcName with
                | "printf" | "scanf" | "puts" ->
                    Some (sprintf "  %s = call i32 @%s(%s)" result funcName args)
                | _ ->
                    Some (sprintf "  %s = call %s @%s(%s)" result returnType funcName args)
            else
                // Handle void calls
                let voidCallPattern = @"func\.call\s*@(\w+)\(([^)]*)\)"
                let vm = Regex.Match(trimmed, voidCallPattern)
                if vm.Success then
                    Some (sprintf "  call void @%s(%s)" vm.Groups.[1].Value vm.Groups.[2].Value)
                else None
        
        elif trimmed.StartsWith("func.return") then
            let returnPattern = @"func\.return\s*([^:]*)?(?::\s*(.+))?"
            let m = Regex.Match(trimmed, returnPattern)
            if m.Success && not (String.IsNullOrWhiteSpace(m.Groups.[1].Value)) then
                let value = m.Groups.[1].Value.Trim()
                let returnType = if m.Groups.[2].Success then m.Groups.[2].Value else "i32"
                Some (sprintf "  ret %s %s" returnType value)
            else
                Some "  ret i32 0"
        
        elif trimmed.StartsWith("memref.get_global") then
            let getGlobalPattern = @"(%\w+)\s*=\s*memref\.get_global\s*(@\w+)"
            let m = Regex.Match(trimmed, getGlobalPattern)
            if m.Success then
                let result = m.Groups.[1].Value
                let global = m.Groups.[2].Value
                Some (sprintf "  %s = getelementptr inbounds [256 x i8], [256 x i8]* %s, i64 0, i64 0" result global)
            else None
        
        elif trimmed.StartsWith("memref.alloca") then
            let allocaPattern = @"(%\w+)\s*=\s*memref\.alloca\(\)"
            let m = Regex.Match(trimmed, allocaPattern)
            if m.Success then
                Some (sprintf "  %s = alloca [256 x i8], align 1" m.Groups.[1].Value)
            else None
        
        elif trimmed.EndsWith(":") then
            // Basic block label
            Some trimmed
        
        else
            None
    
    /// Converts MLIR function to LLVM IR
    let convertMLIRFunction (func: ParsedMLIRFunction) : string =
        let paramStr = 
            if func.Parameters.IsEmpty then ""
            else
                func.Parameters
                |> List.mapi (fun i (name, ptype) -> 
                    let llvmType = match ptype with
                                   | "i32" -> "i32"
                                   | "i64" -> "i64"
                                   | _ -> "i32"
                    sprintf "%s %%%s" llvmType name)
                |> String.concat ", "
        
        let llvmReturnType = match func.ReturnType with
                            | "i32" -> "i32"
                            | "i64" -> "i64"  
                            | "void" -> "void"
                            | _ -> "i32"
        
        if func.IsExternal then
            sprintf "declare %s @%s(%s)" llvmReturnType func.Name paramStr
        else
            let header = 
                if func.Name = "main" then
                    sprintf "define i32 @main(i32 %%argc, i8** %%argv) {"
                else
                    sprintf "define %s @%s(%s) {" llvmReturnType func.Name paramStr
            
            let body = 
                func.Body
                |> List.choose convertMLIRInstruction
                |> List.filter (not << String.IsNullOrWhiteSpace)
            
            let footer = "}"
            
            [header] @ body @ [footer] |> String.concat "\n"
    
    /// Converts MLIR global to LLVM IR
    let convertMLIRGlobal (global: ParsedMLIRGlobal) : string =
        // Parse memref type and dense value
        let typePattern = @"memref<(\d+)x\s*i8>"
        let tm = Regex.Match(global.Type, typePattern)
        
        let valuePattern = @"dense<""([^""]*)"">?"
        let vm = Regex.Match(global.Value, valuePattern)
        
        if tm.Success && vm.Success then
            let size = tm.Groups.[1].Value
            let content = vm.Groups.[1].Value.Replace("\\00", "")
            let actualSize = content.Length + 1
            sprintf "%s = private unnamed_addr constant [%d x i8] c\"%s\\00\", align 1" 
                    global.Name actualSize content
        else
            sprintf "%s = private unnamed_addr constant [1 x i8] zeroinitializer, align 1" global.Name
    
    /// Main MLIR to LLVM transformation
    let transformMLIRToLLVM (mlirText: string) (targetTriple: string) : CompilerResult<LLVMOutput> =
        if String.IsNullOrWhiteSpace(mlirText) then
            CompilerFailure [TransformError("MLIR to LLVM", "empty input", "LLVM IR", "Input MLIR is empty")]
        else
            try
                printfn "Debug: Starting MLIR to LLVM transformation"
                printfn "Debug: MLIR input length: %d characters" mlirText.Length
                
                let (functions, globals) = parseMLIRModule mlirText
                
                printfn "Debug: Parsed %d functions and %d globals" functions.Length globals.Length
                functions |> List.iter (fun f -> printfn "  Function: %s (external: %b)" f.Name f.IsExternal)
                
                // Extract module name from MLIR
                let moduleNamePattern = @"module\s+([^\s{]+)"
                let moduleMatch = Regex.Match(mlirText, moduleNamePattern)
                let moduleName = 
                    if moduleMatch.Success then moduleMatch.Groups.[1].Value
                    else "main"
                
                // Generate LLVM IR components
                let moduleHeader = [
                    sprintf "; ModuleID = '%s'" moduleName
                    sprintf "source_filename = \"%s\"" moduleName
                    "target datalayout = \"e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128\""
                    sprintf "target triple = \"%s\"" targetTriple
                    ""
                ]
                
                // Convert globals
                let globalDeclarations = 
                    globals 
                    |> List.map convertMLIRGlobal
                
                // External function declarations for I/O
                let standardDeclarations = [
                    "declare i32 @printf(i8*, ...)"
                    "declare i32 @scanf(i8*, ...)"
                    "declare i32 @puts(i8*)"
                    "declare i32 @getchar()"
                    ""
                ]
                
                // Convert functions
                let functionDefinitions = 
                    functions
                    |> List.map convertMLIRFunction
                
                let completeIR = 
                    moduleHeader @ 
                    globalDeclarations @
                    [""] @
                    standardDeclarations @
                    functionDefinitions @
                    [""]
                    |> String.concat "\n"
                
                printfn "Debug: Generated LLVM IR length: %d characters" completeIR.Length
                
                Success {
                    ModuleName = moduleName
                    LLVMIRText = completeIR
                    SymbolTable = Map.empty
                    ExternalFunctions = ["printf"; "scanf"; "puts"; "getchar"]
                    GlobalVariables = globals |> List.map (fun g -> g.Name)
                }
            
            with
            | ex ->
                printfn "Debug: Exception in MLIR to LLVM transformation: %s" ex.Message
                CompilerFailure [TransformError("MLIR to LLVM", "MLIR parsing", "LLVM IR", ex.Message)]

/// Main translation entry point
let translateToLLVM (mlirText: string) : CompilerResult<LLVMOutput> =
    let targetTriple = TargetTripleManagement.getTargetTriple "default"
    TargetTripleManagement.validateTargetTriple targetTriple >>= fun _ ->
    EnhancedMLIRProcessing.transformMLIRToLLVM mlirText targetTriple

/// External tool integration for native compilation
module ExternalToolchain =
    
    /// Enhanced command availability check for MSYS2/Windows
    let isCommandAvailable (command: string) : bool =
        try
            let commands = 
                if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
                    [command; command + ".exe"]
                else
                    [command]
            
            commands |> List.exists (fun cmd ->
                try
                    let processInfo = System.Diagnostics.ProcessStartInfo()
                    processInfo.FileName <- cmd
                    processInfo.Arguments <- "--version"
                    processInfo.UseShellExecute <- false
                    processInfo.RedirectStandardOutput <- true
                    processInfo.RedirectStandardError <- true
                    processInfo.CreateNoWindow <- true
                    
                    use proc = System.Diagnostics.Process.Start(processInfo)
                    proc.WaitForExit(5000) |> ignore
                    proc.ExitCode = 0
                with _ -> false)
        with _ -> false
    
    /// Enhanced compiler command detection
    let getCompilerCommands (target: string) : CompilerResult<string * string> =
        if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
            let llcOptions = ["llc"; "llc.exe"]
            let linkerOptions = ["gcc"; "gcc.exe"; "clang"; "clang.exe"]
            
            let llcFound = llcOptions |> List.tryFind isCommandAvailable
            let linkerFound = linkerOptions |> List.tryFind isCommandAvailable
            
            match llcFound, linkerFound with
            | Some llc, Some linker ->
                Success (llc, linker)
            | None, _ ->
                CompilerFailure [CompilerError("toolchain", "LLVM compiler (llc) not found", 
                    Some "Install LLVM tools: pacman -S mingw-w64-x86_64-llvm")]
            | _, None ->
                CompilerFailure [CompilerError("toolchain", "C compiler not found", 
                    Some "Install GCC: pacman -S mingw-w64-x86_64-gcc")]
        else
            if isCommandAvailable "llc" && isCommandAvailable "clang" then
                Success ("llc", "clang")
            elif isCommandAvailable "llc" && isCommandAvailable "gcc" then
                Success ("llc", "gcc")
            else
                CompilerFailure [CompilerError("toolchain", "Required compilers not found", 
                    Some "Install LLVM and GCC/Clang")]
    
    /// Enhanced external command execution with better error handling
    let runExternalCommand (command: string) (arguments: string) (workingDir: string option) : CompilerResult<string> =
        try
            printfn "Debug: Executing %s %s" command arguments
            
            let processInfo = System.Diagnostics.ProcessStartInfo()
            processInfo.FileName <- command
            processInfo.Arguments <- arguments
            processInfo.UseShellExecute <- false
            processInfo.RedirectStandardOutput <- true
            processInfo.RedirectStandardError <- true
            processInfo.CreateNoWindow <- true
            
            match workingDir with
            | Some dir -> processInfo.WorkingDirectory <- dir
            | None -> ()
            
            use proc = System.Diagnostics.Process.Start(processInfo)
            let output = proc.StandardOutput.ReadToEnd()
            let error = proc.StandardError.ReadToEnd()
            proc.WaitForExit()
            
            printfn "Debug: Command exit code: %d" proc.ExitCode
            if not (String.IsNullOrWhiteSpace(output)) then
                printfn "Debug: Command output: %s" output
            if not (String.IsNullOrWhiteSpace(error)) then
                printfn "Debug: Command error: %s" error
            
            if proc.ExitCode = 0 then
                Success output
            else
                CompilerFailure [CompilerError("external command", 
                    sprintf "%s failed with exit code %d" command proc.ExitCode, 
                    Some error)]
        with
        | ex ->
            CompilerFailure [CompilerError("external command", 
                sprintf "Failed to execute %s" command, 
                Some ex.Message)]

/// Enhanced native compilation with better MSYS2 support
let compileLLVMToNative (llvmOutput: LLVMOutput) (outputPath: string) (target: string) : CompilerResult<unit> =
    let targetTriple = TargetTripleManagement.getTargetTriple target
    TargetTripleManagement.validateTargetTriple targetTriple >>= fun _ ->
    
    ExternalToolchain.getCompilerCommands target >>= fun (llcCommand, linkerCommand) ->
    
    let llvmPath = Path.ChangeExtension(outputPath, ".ll")
    let objPath = Path.ChangeExtension(outputPath, ".o")
    let outputDir = Path.GetDirectoryName(outputPath)
    
    try
        // Write LLVM IR to file
        let utf8WithoutBom = System.Text.UTF8Encoding(false)
        File.WriteAllText(llvmPath, llvmOutput.LLVMIRText, utf8WithoutBom)
        printfn "Saved LLVM IR to: %s" llvmPath
        
        // Compile to object file with enhanced options
        let llcArgs = sprintf "-filetype=obj -mtriple=%s -relocation-model=pic -o \"%s\" \"%s\"" 
                             targetTriple objPath llvmPath
        printfn "Compiling to native code for target '%s'..." targetTriple
        
        ExternalToolchain.runExternalCommand llcCommand llcArgs (Some outputDir) >>= fun _ ->
        printfn "Created object file: %s" objPath
        
        // Enhanced linking with proper Windows runtime libraries
        let linkArgs = 
            if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
                sprintf "\"%s\" -o \"%s\" -mconsole -static-libgcc -lmsvcrt -lkernel32" 
                        objPath outputPath
            else
                sprintf "\"%s\" -o \"%s\"" objPath outputPath
        
        printfn "Linking object file to executable..."        
        ExternalToolchain.runExternalCommand linkerCommand linkArgs (Some outputDir) >>= fun linkOutput ->
        
        // Verify output file exists
        if File.Exists(outputPath) then
            printfn "Successfully created executable: %s" outputPath
            // Cleanup intermediate files but keep LLVM IR for debugging
            if File.Exists(objPath) then File.Delete(objPath)
            Success ()
        else
            CompilerFailure [CompilerError(
                "native compilation", 
                "Executable was not created", 
                Some (sprintf "Linker output: %s" linkOutput))]
    
    with
    | ex ->
        CompilerFailure [CompilerError("native compilation", 
            "Failed during native compilation", 
            Some ex.Message)]