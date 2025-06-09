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

/// MLIR parsing state and function types
type MLIRParsingState = {
    CurrentModule: string
    Functions: Map<string, MLIRFunction>
    Globals: Map<string, MLIRGlobal>
    ErrorContext: string list
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

/// MLIR parsing combinators
module MLIRParsers =
    
    /// Parses a string token
    let ptoken (token: string) : Parser<string, MLIRParsingState> =
        pstring token |>> (fun _ -> token)
    
    /// Parses a valid MLIR identifier
    let identifier : Parser<string, MLIRParsingState> =
        let isIdStart c = isLetter c || c = '_' || c = '@' || c = '%'
        let isIdCont c = isLetter c || isDigit c || c = '_' || c = '.'
        
        satisfy isIdStart >>= fun first ->
        many (satisfy isIdCont) >>= fun rest ->
        let idStr = String(Array.ofList (first :: rest))
        succeed idStr
        |> withErrorContext "MLIR identifier"
    
    /// Parses whitespace including comments
    let whitespace : Parser<unit, MLIRParsingState> =
        let comment = 
            pstring "//" >>= fun _ -> 
            many (satisfy (fun c -> c <> '\n')) >>= fun _ ->
            succeed ()
        
        many (ws <|> comment) |>> ignore
        |> withErrorContext "whitespace"
    
    /// Parses a type definition
    let typeParser : Parser<string, MLIRParsingState> =
        let primitiveType = 
            choice [
                pstring "i1"
                pstring "i8"
                pstring "i16"
                pstring "i32"
                pstring "i64"
                pstring "f32"
                pstring "f64"
                pstring "void"
            ]
        
        let memrefType =
            pstring "memref<" >>= fun _ ->
            many (satisfy (fun c -> c <> '>')) >>= fun content ->
            pchar '>' >>= fun _ ->
            let contentStr = String(Array.ofList content)
            succeed ("memref<" + contentStr + ">")
        
        primitiveType <|> memrefType
        |> withErrorContext "MLIR type"
    
    /// Parses a function parameter
    let parameterParser : Parser<string * string, MLIRParsingState> =
        identifier >>= fun name ->
        whitespace >>= fun _ ->
        pchar ':' >>= fun _ ->
        whitespace >>= fun _ ->
        typeParser >>= fun paramType ->
        succeed (name, paramType)
        |> withErrorContext "function parameter"
    
    /// Parses a list of parameters
    let parameterListParser : Parser<(string * string) list, MLIRParsingState> =
        sepBy parameterParser (pchar ',' >>= fun _ -> whitespace)
        |> withErrorContext "parameter list"
    
    /// Parses a function signature
    let functionSignatureParser : Parser<MLIRFunction, MLIRParsingState> =
        let isPrivate = 
            opt (ptoken "private" >>= fun _ -> whitespace) >>= fun isPrivateOpt ->
            succeed (isPrivateOpt.IsSome)
        
        ptoken "func.func" >>= fun _ ->
        whitespace >>= fun _ ->
        isPrivate >>= fun isExternal ->
        identifier >>= fun name ->
        whitespace >>= fun _ ->
        between (pchar '(') (pchar ')') parameterListParser >>= fun parameters ->
        whitespace >>= fun _ ->
        opt (pstring "->" >>= fun _ -> whitespace >>= fun _ -> typeParser) >>= fun returnTypeOpt ->
        
        let returnType = defaultArg returnTypeOpt "void"
        succeed {
            Name = name
            Parameters = parameters
            ReturnType = returnType
            Body = []
            IsExternal = isExternal
        }
        |> withErrorContext "function signature"
    
    /// Parses a global constant definition
    let globalConstantParser : Parser<MLIRGlobal, MLIRParsingState> =
        ptoken "memref.global" >>= fun _ ->
        whitespace >>= fun _ ->
        opt (ptoken "constant" >>= fun _ -> whitespace) >>= fun isConstantOpt ->
        identifier >>= fun name ->
        whitespace >>= fun _ ->
        pchar ':' >>= fun _ ->
        whitespace >>= fun _ ->
        typeParser >>= fun globalType ->
        whitespace >>= fun _ ->
        pchar '=' >>= fun _ ->
        whitespace >>= fun _ ->
        many (satisfy (fun c -> c <> '\n')) >>= fun valueChars ->
        
        let value = String(Array.ofList valueChars)
        succeed {
            Name = name
            Type = globalType
            Value = value
            IsConstant = isConstantOpt.IsSome
        }
        |> withErrorContext "global constant"

/// Type conversion from MLIR to LLVM
module TypeConversion =
    
    /// Converts MLIR type to LLVM type
    let mlirTypeToLLVM (mlirType: string) : string =
        match mlirType with
        | "i1" | "i8" | "i16" | "i32" | "i64" -> mlirType
        | "f32" -> "float"
        | "f64" -> "double"
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
                "i8*"  // Default pointer type
        | _ -> "i32"  // Default integer type

/// Function body parsing and conversion
module FunctionBodyParsers =
    
    /// Parses a single MLIR operation and converts to LLVM
    let operationParser : Parser<string, MLIRParsingState> =
        let arithConstant =
            pstring "arith.constant" >>= (fun _ ->
                many (satisfy (fun c -> c <> '\n')) >>= (fun rest ->
                    let restStr = String(Array.ofList rest)
                    succeed (restStr.Replace("arith.constant", "add"))
                )
            )
        
        let funcCall = 
            pstring "func.call" >>= (fun _ ->
                many (satisfy (fun c -> c <> '\n')) >>= (fun rest ->
                    let restStr = String(Array.ofList rest)
                    succeed (restStr.Replace("func.call", "call"))
                )
            )
        
        let funcReturn =
            pstring "func.return" >>= (fun _ ->
                many (satisfy (fun c -> c <> '\n')) >>= (fun rest ->
                    let restStr = String(Array.ofList rest)
                    succeed (restStr.Replace("func.return", "ret"))
                )
            )
        
        let memrefGet =
            pstring "memref.get_global" >>= (fun _ ->
                many (satisfy (fun c -> c <> '\n')) >>= (fun rest ->
                    let restStr = String(Array.ofList rest)
                    succeed (restStr.Replace("memref.get_global", "getelementptr"))
                )
            )
        
        let memrefAlloca =
            pstring "memref.alloca" >>= (fun _ ->
                many (satisfy (fun c -> c <> '\n')) >>= (fun rest ->
                    let restStr = String(Array.ofList rest)
                    succeed (restStr.Replace("memref.alloca", "alloca"))
                )
            )
        
        choice [
            arithConstant
            funcCall
            funcReturn
            memrefGet
            memrefAlloca
            many (satisfy (fun c -> c <> '\n')) |>> String.Concat
        ]
        |> withErrorContext "MLIR operation"

/// Enhanced MLIR processing using XParsec combinators
module MLIRProcessing =
    
    /// Extracts module name from MLIR text
    let extractModuleName : Parser<string, MLIRParsingState> =
        let moduleNameParser = 
            pstring "module" >>= (fun _ -> 
                whitespace >>= (fun _ ->
                    identifier
                )
            )
        
        opt moduleNameParser >>= (fun nameOpt ->
            match nameOpt with
            | Some name -> succeed name
            | None -> succeed "main"
        )
        |> withErrorContext "module name extraction"
    
    /// Processes a line of MLIR code
    let processMLIRLine (line: string) : Parser<unit, MLIRParsingState> =
        let trimmedLine = line.Trim()
        
        let lineParser =
            if String.IsNullOrWhiteSpace(trimmedLine) || trimmedLine.StartsWith("//") then
                succeed ()
            elif trimmedLine.StartsWith("module") then
                extractModuleName >>= (fun moduleName ->
                    fun state ->
                        let newState = { state with CurrentModule = moduleName }
                        Reply(Ok (), newState)
                )
            elif trimmedLine.StartsWith("func.func") then
                MLIRParsers.functionSignatureParser >>= (fun func ->
                    fun state ->
                        let newFunctions = Map.add func.Name func state.Functions
                        let newState = { state with Functions = newFunctions }
                        Reply(Ok (), newState)
                )
            elif trimmedLine.StartsWith("memref.global") then
                MLIRParsers.globalConstantParser >>= (fun ``global`` ->
                    fun state ->
                        let newGlobals = Map.add ``global``.Name ``global`` state.Globals
                        let newState = { state with Globals = newGlobals }
                        Reply(Ok (), newState)
                )
            else
                succeed ()
                
        lineParser |> withErrorContext "MLIR line processing"
    
    /// Processes complete MLIR module text
    let processMLIRModule (mlirText: string) : CompilerResult<MLIRParsingState> =
        let lines = mlirText.Split('\n')
        let initialState = {
            CurrentModule = "main"
            Functions = Map.empty
            Globals = Map.empty
            ErrorContext = []
        }
        
        let rec processLines state lineIndex =
            if lineIndex >= lines.Length then
                Success state
            else
                let line = lines.[lineIndex]
                match processMLIRLine line state with
                | Reply(Ok (), newState) -> processLines newState (lineIndex + 1)
                | Reply(Error, errorMsg) -> 
                    CompilerFailure [TransformError(
                        "MLIR processing", 
                        sprintf "line %d" lineIndex, 
                        "processed MLIR", 
                        errorMsg)]
        
        processLines initialState 0

/// LLVM IR generation from parsed MLIR
module LLVMGeneration =
    
    /// Converts MLIR function to LLVM IR
    let convertFunction (func: MLIRFunction) : string =
        let paramStr = 
            if func.Parameters.IsEmpty then ""
            else
                func.Parameters
                |> List.map (fun (name, paramType) -> 
                    let llvmType = TypeConversion.mlirTypeToLLVM paramType
                    sprintf "%s %%%s" llvmType name)
                |> String.concat ", "
        
        let llvmReturnType = TypeConversion.mlirTypeToLLVM func.ReturnType
        
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
                |> List.map (fun line -> "  " + line)
                |> String.concat "\n"
            
            let footer = "}"
            
            [header; body; footer] |> String.concat "\n"
    
    /// Converts MLIR global to LLVM IR
    let convertGlobal (``global``: MLIRGlobal) : string =
        // Parse the type and value 
        let typeMatch = ``global``.Type.Contains("memref<")
        let valueMatch = ``global``.Value.Contains("dense<")
        
        if typeMatch && valueMatch then
            // Extract size from memref<NxiM>
            let sizeStart = ``global``.Type.IndexOf('<') + 1
            let sizeEnd = ``global``.Type.IndexOf('x', sizeStart)
            let size = 
                if sizeEnd > sizeStart then
                    ``global``.Type.Substring(sizeStart, sizeEnd - sizeStart)
                else "1"
            
            // Extract content from dense<"value">
            let contentStart = ``global``.Value.IndexOf('"') + 1
            let contentEnd = ``global``.Value.LastIndexOf('"')
            let content = 
                if contentEnd > contentStart then
                    ``global``.Value.Substring(contentStart, contentEnd - contentStart)
                       .Replace("\\00", "")
                else ""
            
            let actualSize = content.Length + 1  // Include null terminator
            sprintf "%s = private unnamed_addr constant [%d x i8] c\"%s\\00\", align 1" 
                    ``global``.Name actualSize content
        else
            sprintf "%s = private unnamed_addr constant [1 x i8] zeroinitializer, align 1" ``global``.Name
    
    /// Generates LLVM IR module from parsed MLIR
    let generateLLVMModule (state: MLIRParsingState) (targetTriple: string) : string =
        let moduleHeader = [
            sprintf "; ModuleID = '%s'" state.CurrentModule
            sprintf "source_filename = \"%s\"" state.CurrentModule
            "target datalayout = \"e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128\""
            sprintf "target triple = \"%s\"" targetTriple
            ""
        ]
        
        let globalDeclarations = 
            state.Globals
            |> Map.toList
            |> List.map (snd >> convertGlobal)
        
        // Standard declarations for I/O
        let standardDeclarations = [
            "declare i32 @printf(i8*, ...)"
            "declare i32 @scanf(i8*, ...)"
            "declare i32 @puts(i8*)"
            "declare i32 @getchar()"
            ""
        ]
        
        let functionDefinitions = 
            state.Functions
            |> Map.toList
            |> List.map (snd >> convertFunction)
        
        let completeIR = 
            moduleHeader @ 
            globalDeclarations @
            [""] @
            standardDeclarations @
            functionDefinitions @
            [""]
            |> String.concat "\n"
        
        completeIR

/// Main MLIR to LLVM transformation
let transformMLIRToLLVM (mlirText: string) (targetTriple: string) : CompilerResult<LLVMOutput> =
    if String.IsNullOrWhiteSpace(mlirText) then
        CompilerFailure [TransformError("MLIR to LLVM", "empty input", "LLVM IR", "Input MLIR is empty")]
    else
        try
            MLIRProcessing.processMLIRModule mlirText >>= fun state ->
            let llvmIR = LLVMGeneration.generateLLVMModule state targetTriple
            
            let externalFunctions = ["printf"; "scanf"; "puts"; "getchar"]
            let globalVariables = state.Globals |> Map.toList |> List.map fst
            
            Success {
                ModuleName = state.CurrentModule
                LLVMIRText = llvmIR
                SymbolTable = Map.empty
                ExternalFunctions = externalFunctions
                GlobalVariables = globalVariables
            }
        with
        | ex ->
            CompilerFailure [TransformError("MLIR to LLVM", "MLIR parsing", "LLVM IR", ex.Message)]

/// Main translation entry point
let translateToLLVM (mlirText: string) : CompilerResult<LLVMOutput> =
    let targetTriple = TargetTripleManagement.getTargetTriple "default"
    TargetTripleManagement.validateTargetTriple targetTriple >>= fun _ ->
    transformMLIRToLLVM mlirText targetTriple

/// External tool integration for native compilation
module ExternalToolchain =
    
    /// Checks if a command is available
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
    
    /// Gets compiler commands for the target platform
    let getCompilerCommands (target: string) : CompilerResult<string * string> =
        if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
            let llcOptions = ["llc"; "llc.exe"]
            let linkerOptions = ["gcc"; "gcc.exe"; "clang"; "clang.exe"]
            
            let llcFound = llcOptions |> List.tryFind isCommandAvailable
            let linkerFound = linkerOptions |> List.tryFind isCommandAvailable
            
            match llcFound, linkerFound with
            | Some llc, Some linker -> Success (llc, linker)
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
    
    /// Runs an external command with error handling
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
                CompilerFailure [CompilerError("external command", 
                    sprintf "%s failed with exit code %d" command proc.ExitCode, 
                    Some error)]
        with ex ->
            CompilerFailure [CompilerError("external command", 
                sprintf "Failed to execute %s" command, 
                Some ex.Message)]

/// Compiles LLVM IR to native code
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
        
        // Compile to object file
        let llcArgs = sprintf "-filetype=obj -mtriple=%s -relocation-model=pic -o \"%s\" \"%s\"" 
                             targetTriple objPath llvmPath
        
        ExternalToolchain.runExternalCommand llcCommand llcArgs (Some outputDir) >>= fun _ ->
        
        // Link the final executable
        let linkArgs = 
            if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
                sprintf "\"%s\" -o \"%s\" -mconsole -static-libgcc -lmsvcrt -lkernel32" 
                        objPath outputPath
            else
                sprintf "\"%s\" -o \"%s\"" objPath outputPath
        
        ExternalToolchain.runExternalCommand linkerCommand linkArgs (Some outputDir) >>= fun _ ->
        
        // Verify output file exists
        if File.Exists(outputPath) then
            if File.Exists(objPath) then File.Delete(objPath)
            Success ()
        else
            CompilerFailure [CompilerError("native compilation", 
                "Executable was not created", 
                Some "Linking failed to produce output file")]
    
    with
    | ex ->
        CompilerFailure [CompilerError("native compilation", 
            "Failed during native compilation", 
            Some ex.Message)]

/// Validates that the LLVM IR has no heap allocations
let validateZeroAllocationGuarantees (llvmIR: string) : CompilerResult<unit> =
    // Check for common heap allocation functions
    let heapFunctions = ["malloc"; "calloc"; "realloc"; "new"]
    
    let containsHeapAllocation = 
        heapFunctions
        |> List.exists (fun func -> 
            let pattern = sprintf "call.*@%s" func
            llvmIR.Contains(pattern))
    
    if containsHeapAllocation then
        CompilerFailure [TransformError(
            "zero-allocation validation", 
            "optimized LLVM IR", 
            "zero-allocation LLVM IR", 
            "Found potential heap allocation functions in LLVM IR")]
    else
        Success ()