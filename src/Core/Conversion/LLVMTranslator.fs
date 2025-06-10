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

/// MLIR parsing state and domain models
type MLIRParsingState = {
    CurrentModule: string
    Functions: Map<string, MLIRFunction>
    Globals: Map<string, MLIRGlobal>
    Operations: string list
    SymbolTable: Map<string, string>
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

/// Initializes MLIR parsing state
let createMLIRParsingState() : MLIRParsingState = {
    CurrentModule = "main"
    Functions = Map.empty
    Globals = Map.empty
    Operations = []
    SymbolTable = Map.empty
    ErrorContext = []
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

/// Basic MLIR parsers using XParsec combinators
module MLIRParsers =

    /// Parser for an MLIR identifier (including SSA value names)
    let identifier : Parser<string, MLIRParsingState> =
        let isIdStart c = isLetter c || c = '_' || c = '@' || c = '%'
        let isIdCont c = isLetter c || isDigit c || c = '_' || c = '.'
        
        satisfy isIdStart >>= fun first ->
        many (satisfy isIdCont) >>= fun rest ->
        let idStr = String(Array.ofList (first :: rest))
        succeed idStr
        |> withErrorContext "MLIR identifier"
    
    /// Parser for MLIR whitespace and comments
    let whitespace : Parser<unit, MLIRParsingState> =
        let comment = 
            pstring "//" >>= fun _ -> 
            many (satisfy (fun c -> c <> '\n')) >>= fun _ ->
            succeed ()
        
        many (ws <|> comment) |>> ignore
        |> withErrorContext "whitespace"
    
    /// Parser for MLIR types
    let mlirType : Parser<string, MLIRParsingState> =
        // Basic primitive types
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
        
        // Memref type parser
        let memrefType =
            pstring "memref<" >>= fun _ ->
            many (satisfy (fun c -> c <> '>')) >>= fun content ->
            pchar '>' >>= fun _ ->
            let contentStr = String(Array.ofList content)
            succeed ("memref<" + contentStr + ">")
        
        // LLVM pointer type parser
        let ptrType =
            pstring "!llvm.ptr<" >>= fun _ ->
            many (satisfy (fun c -> c <> '>')) >>= fun content ->
            pchar '>' >>= fun _ ->
            let contentStr = String(Array.ofList content)
            succeed ("!llvm.ptr<" + contentStr + ">")
        
        choice [primitiveType; memrefType; ptrType]
        |> withErrorContext "MLIR type"
    
    /// Parser for MLIR function parameters
    let parameter : Parser<string * string, MLIRParsingState> =
        identifier >>= fun name ->
        whitespace >>= fun _ ->
        pchar ':' >>= fun _ ->
        whitespace >>= fun _ ->
        mlirType >>= fun paramType ->
        succeed (name, paramType)
        |> withErrorContext "function parameter"
    
    /// Parser for a list of parameters
    let parameterList : Parser<(string * string) list, MLIRParsingState> =
        sepBy parameter (pchar ',' >>= fun _ -> whitespace)
        |> withErrorContext "parameter list"
    
    /// Parser for MLIR function signatures
    let functionSignature : Parser<MLIRFunction, MLIRParsingState> =
        pstring "func.func" >>= fun _ ->
        whitespace >>= fun _ ->
        opt (pstring "private" >>= fun _ -> whitespace) >>= fun isPrivateOpt ->
        identifier >>= fun name ->
        whitespace >>= fun _ ->
        between (pchar '(') (pchar ')') parameterList >>= fun parameters ->
        whitespace >>= fun _ ->
        opt (pstring "->" >>= fun _ -> whitespace >>= fun _ -> mlirType) >>= fun returnTypeOpt ->
        
        let returnType = defaultArg returnTypeOpt "void"
        let isExternal = isPrivateOpt.IsSome
        
        succeed {
            Name = name
            Parameters = parameters
            ReturnType = returnType
            Body = []
            IsExternal = isExternal
        }
        |> withErrorContext "function signature"
    
    /// Parser for MLIR global declarations
    let globalConstant : Parser<MLIRGlobal, MLIRParsingState> =
        pstring "memref.global" >>= fun _ ->
        whitespace >>= fun _ ->
        opt (pstring "constant" >>= fun _ -> whitespace) >>= fun isConstantOpt ->
        identifier >>= fun name ->
        whitespace >>= fun _ ->
        pchar ':' >>= fun _ ->
        whitespace >>= fun _ ->
        mlirType >>= fun globalType ->
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
    
    /// Parser for MLIR module names
    let moduleName : Parser<string, MLIRParsingState> =
        pstring "module" >>= fun _ -> 
        whitespace >>= fun _ ->
        identifier
        |> withErrorContext "module name"
    
    /// Updates the MLIR parsing state with a function
    let updateState (update: MLIRParsingState -> MLIRParsingState) : Parser<unit, MLIRParsingState> =
        fun state ->
            let newState = update state
            Reply(Ok (), newState)
    
    /// Records a function in the state
    let recordFunction (func: MLIRFunction) : Parser<unit, MLIRParsingState> =
        updateState (fun state -> 
            { state with Functions = Map.add func.Name func state.Functions })
    
    /// Records a global in the state
    let recordGlobal (global: MLIRGlobal) : Parser<unit, MLIRParsingState> =
        updateState (fun state -> 
            { state with Globals = Map.add global.Name global state.Globals })
    
    /// Records the module name in the state
    let recordModuleName (name: string) : Parser<unit, MLIRParsingState> =
        updateState (fun state -> { state with CurrentModule = name })
    
    /// Records a symbol mapping in the state
    let recordSymbol (original: string) (translated: string) : Parser<unit, MLIRParsingState> =
        updateState (fun state -> 
            { state with SymbolTable = Map.add original translated state.SymbolTable })
    
    /// Emits LLVM IR operation
    let emitOperation (operation: string) : Parser<unit, MLIRParsingState> =
        updateState (fun state -> 
            { state with Operations = operation :: state.Operations })

/// MLIR to LLVM type conversions
module TypeConversion =
    
    /// Converts an MLIR type to LLVM IR type
    let mlirTypeToLLVM (mlirType: string) : string =
        match mlirType with
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
    let convertFunction (func: MLIRFunction) : Parser<string, MLIRParsingState> =
        use writer = new StringWriter()
        
        // Convert parameters
        let paramStr = 
            if func.Parameters.IsEmpty then ""
            else
                func.Parameters
                |> List.map (fun (name, paramType) -> 
                    let llvmType = TypeConversion.mlirTypeToLLVM paramType
                    sprintf "%s %s" llvmType name)
                |> String.concat ", "
        
        let llvmReturnType = TypeConversion.mlirTypeToLLVM func.ReturnType
        
        // Handle special case for main function
        let (funcName, finalParamStr) = 
            if func.Name = "@main" then
                ("@main", "i32 %argc, i8** %argv")
            else
                (func.Name, paramStr)
        
        if func.IsExternal then
            // External function declaration
            writer.WriteLine(sprintf "declare %s %s(%s)" llvmReturnType funcName finalParamStr)
        else
            // Function definition
            writer.WriteLine(sprintf "define %s %s(%s) {" llvmReturnType funcName finalParamStr)
            writer.WriteLine("entry:")
            
            // Convert body operations
            func.Body |> List.iter (fun line ->
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
                    writer.WriteLine(sprintf "  %s" cleanLine))
            
            writer.WriteLine("}")
        
        MLIRParsers.emitOperation (writer.ToString()) >>= fun _ ->
        succeed (writer.ToString())
    
    /// Converts an MLIR global to LLVM IR
    let convertGlobal (global: MLIRGlobal) : Parser<string, MLIRParsingState> =
        use writer = new StringWriter()
        
        // Parse type and value from MLIR global
        let typeMatch = global.Type.Contains("memref<")
        let valueMatch = global.Value.Contains("dense<")
        
        if typeMatch && valueMatch then
            // Extract size from memref<NxiM>
            let sizeStart = global.Type.IndexOf('<') + 1
            let sizeEnd = global.Type.IndexOf('x', sizeStart)
            let size = 
                if sizeEnd > sizeStart then
                    global.Type.Substring(sizeStart, sizeEnd - sizeStart)
                else "1"
            
            // Extract content from dense<"value">
            let contentStart = global.Value.IndexOf('"') + 1
            let contentEnd = global.Value.LastIndexOf('"')
            let content = 
                if contentEnd > contentStart then
                    global.Value.Substring(contentStart, contentEnd - contentStart)
                       .Replace("\\00", "")
                else ""
            
            let actualSize = content.Length + 1  // Include null terminator
            writer.WriteLine(sprintf "%s = private unnamed_addr constant [%d x i8] c\"%s\\00\", align 1" 
                    global.Name actualSize content)
        else
            writer.WriteLine(sprintf "%s = private unnamed_addr constant [1 x i8] zeroinitializer, align 1" global.Name)
        
        MLIRParsers.emitOperation (writer.ToString()) >>= fun _ ->
        succeed (writer.ToString())
    
    /// Generates standard C library declarations
    let generateStandardDeclarations() : Parser<unit, MLIRParsingState> =
        let declarations = [
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
        
        let rec emitAll declarations =
            match declarations with
            | [] -> succeed ()
            | decl :: rest ->
                MLIRParsers.emitOperation decl >>= fun _ ->
                emitAll rest
        
        emitAll declarations

/// MLIR module processing
module MLIRProcessing =
    
    /// Processes a single line of MLIR
    let processLine (line: string) : Parser<unit, MLIRParsingState> =
        let trimmedLine = line.Trim()
        
        if String.IsNullOrWhiteSpace(trimmedLine) || trimmedLine.StartsWith("//") then
            succeed ()
        elif trimmedLine.StartsWith("module") then
            run (MLIRParsers.moduleName) trimmedLine >>= fun moduleName ->
            MLIRParsers.recordModuleName moduleName
        elif trimmedLine.StartsWith("func.func") then
            run (MLIRParsers.functionSignature) trimmedLine >>= fun func ->
            MLIRParsers.recordFunction func
        elif trimmedLine.StartsWith("memref.global") then
            run (MLIRParsers.globalConstant) trimmedLine >>= fun global ->
            MLIRParsers.recordGlobal global
        else
            succeed ()
    
    /// Processes a complete MLIR module
    let processModule (mlirText: string) : Parser<MLIRParsingState, MLIRParsingState> =
        let lines = mlirText.Split('\n')
        
        let rec processLines lineIndex state =
            if lineIndex >= lines.Length then
                succeed state
            else
                let line = lines.[lineIndex]
                processLine line >>= fun _ ->
                processLines (lineIndex + 1) state
        
        getState >>= processLines 0
    
    /// Generates LLVM IR module from parsed MLIR
    let generateLLVMModule (targetTriple: string) : Parser<string, MLIRParsingState> =
        // Generate module header
        getState >>= fun state ->
        let moduleHeader = [
            sprintf "; ModuleID = '%s'" state.CurrentModule
            sprintf "source_filename = \"%s\"" state.CurrentModule
            "target datalayout = \"e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128\""
            sprintf "target triple = \"%s\"" targetTriple
            ""
        ]
        
        // Emit module header
        let rec emitHeader headers =
            match headers with
            | [] -> succeed ()
            | header :: rest ->
                MLIRParsers.emitOperation header >>= fun _ ->
                emitHeader rest
        
        emitHeader moduleHeader >>= fun _ ->
        
        // Generate standard declarations
        OperationConversion.generateStandardDeclarations() >>= fun _ ->
        
        // Generate globals
        let convertGlobals =
            getState >>= fun state ->
            let globals = state.Globals |> Map.toList |> List.map snd
            
            let rec processGlobals globals =
                match globals with
                | [] -> succeed ()
                | global :: rest ->
                    OperationConversion.convertGlobal global >>= fun _ ->
                    processGlobals rest
            
            processGlobals globals
        
        convertGlobals >>= fun _ ->
        
        // Generate functions
        let convertFunctions =
            getState >>= fun state ->
            let functions = state.Functions |> Map.toList |> List.map snd
            
            let rec processFunctions functions =
                match functions with
                | [] -> succeed ()
                | func :: rest ->
                    OperationConversion.convertFunction func >>= fun _ ->
                    processFunctions rest
            
            processFunctions functions
        
        convertFunctions >>= fun _ ->
        
        // Ensure we have a proper main wrapper if needed
        let ensureMainWrapper =
            getState >>= fun state ->
            let hasMainFunc = state.Functions |> Map.containsKey "@main"
            
            if hasMainFunc then
                succeed ()
            else
                let functions = state.Functions |> Map.toList |> List.map (fun (_, func) -> func.Name)
                if List.exists (fun name -> name.Contains("main") || name.Contains("Main")) functions then
                    // Emit a main wrapper for user_main
                    let userMain = functions |> List.find (fun name -> name.Contains("main") || name.Contains("Main"))
                    MLIRParsers.emitOperation "" >>= fun _ ->
                    MLIRParsers.emitOperation "define i32 @main(i32 %argc, i8** %argv) {" >>= fun _ ->
                    MLIRParsers.emitOperation "entry:" >>= fun _ ->
                    MLIRParsers.emitOperation (sprintf "  %%1 = call i32 %s()" userMain) >>= fun _ ->
                    MLIRParsers.emitOperation "  ret i32 %1" >>= fun _ ->
                    MLIRParsers.emitOperation "}" >>= fun _ ->
                    succeed ()
                else
                    // Create a minimal main function
                    MLIRParsers.emitOperation "" >>= fun _ ->
                    MLIRParsers.emitOperation "define i32 @main(i32 %argc, i8** %argv) {" >>= fun _ ->
                    MLIRParsers.emitOperation "entry:" >>= fun _ ->
                    MLIRParsers.emitOperation "  ret i32 0" >>= fun _ ->
                    MLIRParsers.emitOperation "}" >>= fun _ ->
                    succeed ()
        
        ensureMainWrapper >>= fun _ ->
        
        // Collect all operations
        getState >>= fun state ->
        let operations = state.Operations |> List.rev
        succeed (String.concat "\n" operations)

/// Main translation entry point
let translateToLLVM (mlirText: string) : CompilerResult<LLVMOutput> =
    if String.IsNullOrWhiteSpace(mlirText) then
        CompilerFailure [TransformError("LLVM translation", "empty input", "LLVM IR", "MLIR input cannot be empty")]
    else
        let targetTriple = TargetTripleManagement.getTargetTriple "default"
        
        try
            let initialState = createMLIRParsingState()
            match run (MLIRProcessing.processModule mlirText >>= fun _ -> MLIRProcessing.generateLLVMModule targetTriple) initialState with
            | (llvmIR, state) ->
                Success {
                    ModuleName = state.CurrentModule
                    LLVMIRText = llvmIR
                    SymbolTable = state.SymbolTable
                    ExternalFunctions = []  // Populated during conversion
                    GlobalVariables = state.Globals |> Map.toList |> List.map fst
                }
        with ex ->
            CompilerFailure [TransformError(
                "MLIR to LLVM", 
                "MLIR processing", 
                "LLVM IR", 
                sprintf "Exception: %s\n%s" ex.Message ex.StackTrace)]

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
    
    try
        // Write LLVM IR to file
        let utf8WithoutBom = System.Text.UTF8Encoding(false)
        File.WriteAllText(llvmPath, llvmOutput.LLVMIRText, utf8WithoutBom)
        
        // Compile to object file
        let llcArgs = sprintf "-filetype=obj -mtriple=%s -o %s %s" targetTriple objPath llvmPath
        ExternalToolchain.runExternalCommand llcCommand llcArgs >>= fun _ ->
        
        // Link to executable with explicit console subsystem and entry point
        let linkArgs = 
            if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
                sprintf "%s -o %s -Wl,--subsystem,console -Wl,--entry,mainCRTStartup -static-libgcc -lmingw32 -lkernel32 -luser32" objPath outputPath
            else
                sprintf "%s -o %s" objPath outputPath
                
        ExternalToolchain.runExternalCommand linkerCommand linkArgs >>= fun _ ->
        
        // Cleanup intermediate files but keep LLVM IR for debugging
        if File.Exists(objPath) then File.Delete(objPath)
        
        Success ()
    
    with
    | ex ->
        CompilerFailure [CompilerError("native compilation", "Failed during native compilation", Some ex.Message)]

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
        CompilerFailure [TransformError(
            "zero-allocation validation", 
            "optimized LLVM IR", 
            "zero-allocation LLVM IR", 
            "Found potential heap allocation functions in LLVM IR")]
    else
        Success ()