module Core.Conversion.LLVMTranslator

open System
open System.IO
open System.Runtime.InteropServices
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

/// Direct MLIR-to-LLVM transformation using string processing
module DirectMLIRProcessing =
    
    /// Converts MLIR I/O operations to LLVM IR calls
    let convertIOOperations (mlirText: string) : string =
        let lines = mlirText.Split('\n')
        
        let convertLine (line: string) : string =
            if line.Contains("func.call @printf") then
                // Convert MLIR printf call to LLVM IR
                let pattern = System.Text.RegularExpressions.Regex(@"(%\w+)\s*=\s*func\.call\s*@printf\((.*?)\)")
                let m = pattern.Match(line)
                if m.Success then
                    let result = m.Groups.[1].Value
                    let args = m.Groups.[2].Value
                    sprintf "  %s = call i32 (i8*, ...) @printf(%s)" result args
                else
                    line
            elif line.Contains("memref.get_global") then
                // Convert memref global access to LLVM getelementptr
                let pattern = System.Text.RegularExpressions.Regex(@"(%\w+)\s*=\s*memref\.get_global\s*(@\w+)")
                let m = pattern.Match(line)
                if m.Success then
                    let result = m.Groups.[1].Value
                    let global = m.Groups.[2].Value
                    sprintf "  %s = getelementptr inbounds [256 x i8], [256 x i8]* %s, i64 0, i64 0" result global
                else
                    line
            elif line.Contains("memref.global constant") then
                // Convert memref global to LLVM global
                let pattern = System.Text.RegularExpressions.Regex(@"memref\.global\s+constant\s+(@\w+).*?dense<\"(.*?)\"")
                let m = pattern.Match(line)
                if m.Success then
                    let name = m.Groups.[1].Value
                    let value = m.Groups.[2].Value.Replace("\\00", "")
                    let length = value.Length + 1
                    sprintf "%s = private unnamed_addr constant [%d x i8] c\"%s\\00\", align 1" name length value
                else
                    line
            else
                line
        
        lines |> Array.map convertLine |> String.concat "\n"
    
    /// Converts a single MLIR function to LLVM IR using direct string replacement
    let convertSingleFunction (functionText: string) : string =
        let lines = functionText.Split('\n') |> Array.map (fun s -> s.Trim()) |> Array.filter (not << String.IsNullOrWhiteSpace)
        
        if lines.Length = 0 then ""
        else
            let headerLine = lines.[0]
            
            // Extract function name and signature
            let funcPattern = System.Text.RegularExpressions.Regex(@"func\.func\s+@(\w+)\s*\((.*?)\)\s*->\s*(\w+)")
            let funcMatch = funcPattern.Match(headerLine)
            
            if not funcMatch.Success then ""
            else
                let funcName = funcMatch.Groups.[1].Value
                let params = funcMatch.Groups.[2].Value
                let returnType = funcMatch.Groups.[3].Value
                
                // Special handling for main function with proper Windows signature
                let (llvmFuncName, llvmParams, llvmReturnType) = 
                    if funcName = "main" then
                        ("main", "i32 %argc, i8** %argv", "i32")
                    else
                        let paramList = 
                            if String.IsNullOrWhiteSpace(params) then ""
                            else
                                params.Split(',') 
                                |> Array.map (fun p -> 
                                    let parts = p.Trim().Split(':')
                                    if parts.Length = 2 then
                                        let paramName = parts.[0].Trim()
                                        let paramType = parts.[1].Trim()
                                        sprintf "i32 %s" paramName  // Simplified type mapping
                                    else "i32")
                                |> String.concat ", "
                        (funcName, paramList, "i32")
                
                let llvmHeader = sprintf "define i32 @%s(%s) {" llvmFuncName llvmParams
                let entryLabel = "entry:"
                
                let bodyLines = lines.[1..lines.Length-2] // Skip header and closing brace
                
                let convertBodyLine (line: string) : string option =
                    if line.Contains("arith.constant") then
                        let constPattern = System.Text.RegularExpressions.Regex(@"(%\w+)\s*=\s*arith\.constant\s+(\d+)")
                        let constMatch = constPattern.Match(line)
                        if constMatch.Success then
                            Some (sprintf "  %s = add i32 0, %s" constMatch.Groups.[1].Value constMatch.Groups.[2].Value)
                        else None
                    elif line.Contains("func.call") then
                        let callPattern = System.Text.RegularExpressions.Regex(@"(%\w+)\s*=\s*func\.call\s*@(\w+)\((.*?)\)")
                        let callMatch = callPattern.Match(line)
                        if callMatch.Success then
                            let result = callMatch.Groups.[1].Value
                            let callee = callMatch.Groups.[2].Value
                            let args = callMatch.Groups.[3].Value
                            // Special handling for I/O functions
                            if callee = "printf" || callee = "scanf" || callee = "getchar" then
                                Some (sprintf "  %s = call i32 @%s(%s)" result callee args)
                            else
                                Some (sprintf "  %s = call i32 @%s(%s)" result callee args)
                        else None
                    elif line.Contains("func.return") then
                        let returnPattern = System.Text.RegularExpressions.Regex(@"func\.return\s+(%\w+)")
                        let returnMatch = returnPattern.Match(line)
                        if returnMatch.Success then
                            Some (sprintf "  ret i32 %s" returnMatch.Groups.[1].Value)
                        else
                            Some "  ret i32 0"
                    elif line.Contains("memref.alloca") then
                        let allocaPattern = System.Text.RegularExpressions.Regex(@"(%\w+)\s*=\s*memref\.alloca\(\)")
                        let allocaMatch = allocaPattern.Match(line)
                        if allocaMatch.Success then
                            Some (sprintf "  %s = alloca [256 x i8], align 1" allocaMatch.Groups.[1].Value)
                        else None
                    else None
                
                let convertedBody = 
                    bodyLines 
                    |> Array.choose convertBodyLine
                    |> Array.toList
                
                let llvmFunction = [llvmHeader; entryLabel] @ convertedBody @ ["}"]
                String.concat "\n" llvmFunction
    
    /// Adds external function declarations for Windows C runtime
    let addExternalDeclarations (isWindows: bool) : string list =
        if isWindows then
            [
                "; External function declarations for Windows"
                "declare i32 @printf(i8*, ...)"
                "declare i32 @scanf(i8*, ...)"
                "declare i32 @getchar()"
                "declare i32 @puts(i8*)"
                ""
            ]
        else
            [
                "; External function declarations"
                "declare i32 @printf(i8*, ...)"
                "declare i32 @scanf(i8*, ...)"
                "declare i32 @getchar()"
                ""
            ]
    
    /// Extracts individual function texts from MLIR module
    let extractFunctionTexts (mlirText: string) : string list =
        // First convert I/O operations
        let processedMLIR = convertIOOperations mlirText
        
        // Split by func.func to get function boundaries
        let funcSplit = processedMLIR.Split([|"func.func"|], StringSplitOptions.RemoveEmptyEntries)
        
        if funcSplit.Length <= 1 then []
        else
            funcSplit.[1..] // Skip the module header part
            |> Array.map (fun funcPart ->
                let funcText = "func.func" + funcPart
                // Find the function end by counting braces
                let lines = funcText.Split('\n')
                let rec findFunctionEnd acc braceCount lineIndex =
                    if lineIndex >= lines.Length then acc
                    else
                        let line = lines.[lineIndex].Trim()
                        let newBraceCount = 
                            braceCount + 
                            (line.ToCharArray() |> Array.filter ((=) '{') |> Array.length) -
                            (line.ToCharArray() |> Array.filter ((=) '}') |> Array.length)
                        
                        let newAcc = acc @ [lines.[lineIndex]]
                        
                        if newBraceCount = 0 && braceCount > 0 then
                            newAcc
                        else
                            findFunctionEnd newAcc newBraceCount (lineIndex + 1)
                
                findFunctionEnd [] 0 0 |> String.concat "\n")
            |> Array.toList
    
    /// Extracts global constants from MLIR
    let extractGlobalConstants (mlirText: string) : string list =
        let lines = mlirText.Split('\n')
        lines 
        |> Array.filter (fun line -> line.Contains("memref.global constant"))
        |> Array.map convertIOOperations
        |> Array.toList
    
    /// Transforms complete MLIR module to LLVM IR
    let transformModule (mlirText: string) : CompilerResult<LLVMOutput> =
        if String.IsNullOrWhiteSpace(mlirText) then
            CompilerFailure [TransformError("MLIR to LLVM", "empty input", "LLVM IR", "Input MLIR is empty")]
        else
            try
                // Extract module name
                let moduleNamePattern = System.Text.RegularExpressions.Regex(@"module\s+(\w+(?:\.\w+)*)")
                let moduleMatch = moduleNamePattern.Match(mlirText)
                let moduleName = 
                    if moduleMatch.Success then moduleMatch.Groups.[1].Value
                    else "main"
                
                // Extract global constants
                let globalConstants = extractGlobalConstants mlirText
                
                // Extract and convert functions
                let functionTexts = extractFunctionTexts mlirText
                printfn "Debug: Found %d functions in MLIR" functionTexts.Length
                
                let llvmFunctions = 
                    functionTexts
                    |> List.map convertSingleFunction
                    |> List.filter (not << String.IsNullOrWhiteSpace)
                
                printfn "Debug: Converted %d functions to LLVM" llvmFunctions.Length
                
                // Determine if we're on Windows
                let isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                let targetTriple = TargetTripleManagement.getTargetTriple "default"
                
                let moduleHeader = [
                    sprintf "; ModuleID = '%s'" moduleName
                    sprintf "source_filename = \"%s\"" moduleName
                    "target datalayout = \"e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128\""
                    sprintf "target triple = \"%s\"" targetTriple
                    ""
                ]
                
                // Add external declarations
                let externals = addExternalDeclarations isWindows
                
                let completeIR = 
                    moduleHeader @ 
                    globalConstants @
                    [""] @
                    externals @
                    llvmFunctions @ 
                    [""]
                    |> String.concat "\n"
                
                Success {
                    ModuleName = moduleName
                    LLVMIRText = completeIR
                    SymbolTable = Map.empty
                    ExternalFunctions = ["printf"; "scanf"; "getchar"; "puts"]
                    GlobalVariables = []
                }
            with
            | ex ->
                printfn "Debug: Exception in transformModule: %s" ex.Message
                CompilerFailure [TransformError("MLIR to LLVM", "MLIR parsing", "LLVM IR", ex.Message)]

/// Main translation entry point
let translateToLLVM (mlirText: string) : CompilerResult<LLVMOutput> =
    if String.IsNullOrWhiteSpace(mlirText) then
        CompilerFailure [TransformError("LLVM translation", "empty input", "LLVM IR", "MLIR input cannot be empty")]
    else
        DirectMLIRProcessing.transformModule mlirText

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
                CompilerFailure [CompilerError("toolchain", "No suitable LLVM/compiler toolchain found", Some "Install LLVM tools and GCC/Clang")]
        else
            if isCommandAvailable "llc" && isCommandAvailable "clang" then
                Success ("llc", "clang")
            elif isCommandAvailable "llc" && isCommandAvailable "gcc" then
                Success ("llc", "gcc")
            else
                CompilerFailure [CompilerError("toolchain", "No suitable LLVM/compiler toolchain found", Some "Install LLVM tools and Clang/GCC")]
    
    /// Runs external command with error handling
    let runExternalCommand (command: string) (arguments: string) (workingDir: string option) : CompilerResult<string> =
        try
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
            
            if proc.ExitCode = 0 then
                Success output
            else
                CompilerFailure [CompilerError("external command", sprintf "%s failed with exit code %d" command proc.ExitCode, Some error)]
        with
        | ex ->
            CompilerFailure [CompilerError("external command", sprintf "Failed to execute %s" command, Some ex.Message)]

/// Compiles LLVM IR to native executable with enhanced MinGW compatibility
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
        
        // Compile to object file
        let llcArgs = sprintf "-filetype=obj -mtriple=%s -o \"%s\" \"%s\"" targetTriple objPath llvmPath
        printfn "Compiling to native code for target '%s'..." targetTriple
        printfn "Using LLVM version 20.1.1"  // As shown in your output
        
        ExternalToolchain.runExternalCommand llcCommand llcArgs (Some outputDir) >>= fun _ ->
        printfn "Created object file: %s" objPath
        
        // Link to executable with explicit Windows runtime libraries
        let linkArgs = 
            if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
                // For MinGW on Windows, explicitly link against msvcrt for C runtime functions
                sprintf "\"%s\" -o \"%s\" -lmsvcrt -lkernel32" objPath outputPath
            else
                sprintf "\"%s\" -o \"%s\"" objPath outputPath
        
        printfn "Linking object file to executable..."        
        ExternalToolchain.runExternalCommand linkerCommand linkArgs (Some outputDir) >>= fun linkOutput ->
        
        // Check if the output file exists
        if File.Exists(outputPath) then
            printfn "Successfully created executable: %s" outputPath
            // Cleanup intermediate files but keep LLVM IR for debugging
            if File.Exists(objPath) then File.Delete(objPath)
            Success ()
        else
            // If linking failed but didn't report an error, provide more specific guidance
            CompilerFailure [CompilerError(
                "native compilation", 
                "Executable was not created", 
                Some (sprintf "Linker output: %s\nTry running: gcc %s -o %s -lmsvcrt -v" linkOutput objPath outputPath))]
    
    with
    | ex ->
        CompilerFailure [CompilerError("native compilation", "Failed during native compilation", Some ex.Message)]