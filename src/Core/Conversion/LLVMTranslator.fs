module Core.Conversion.LLVMTranslator

open System
open System.IO
open System.Text.RegularExpressions
open System.Runtime.InteropServices
open Llvm.NET
open Llvm.NET.Values
open Core.MLIRGeneration.Dialect

/// Represents LLVM IR output from the translation process
type LLVMOutput = {
    ModuleName: string
    LLVMIRText: string
    SymbolTable: Map<string, string>
}

/// Extracts module name from MLIR text
let private extractModuleName (mlirText: string) : string =
    let modulePattern = @"module\s+@?(\w+)"
    let match' = Regex.Match(mlirText, modulePattern)
    if match'.Success then
        match'.Groups.[1].Value
    else
        "main"

/// Converts MLIR function signature to LLVM IR function signature
let private convertFunctionSignature (funcName: string) (returnTypeStr: string) : string =
    let returnType = 
        match returnTypeStr.Trim() with
        | "i32" -> "i32"
        | "i64" -> "i64"  
        | "f32" -> "float"
        | "f64" -> "double"
        | "()" | "void" -> "void"
        | _ -> "i32"
    
    sprintf "define %s @%s() {" returnType funcName

/// Converts MLIR constants to LLVM IR constants
let private convertConstants (mlirText: string) : string =
    let constantPattern = @"%(\w+)\s*=\s*arith\.constant\s+(\d+)\s*:\s*(\w+)"
    let matches = Regex.Matches(mlirText, constantPattern)
    
    matches
    |> Seq.cast<Match>
    |> Seq.map (fun match' ->
        let varName = match'.Groups.[1].Value
        let value = match'.Groups.[2].Value
        let mlirType = match'.Groups.[3].Value
        let llvmType = if mlirType = "i32" then "i32" else "i32"
        sprintf "  %%%s = add %s 0, %s" varName llvmType value
    )
    |> String.concat "\n"

/// Converts MLIR function calls to LLVM IR calls
let private convertFunctionCalls (mlirText: string) : string =
    let callPattern = @"%(\w+)\s*=\s*func\.call\s+@(\w+)\s*\(\)\s*:\s*\(\)\s*->\s*(\w+)"
    let matches = Regex.Matches(mlirText, callPattern)
    
    matches
    |> Seq.cast<Match>
    |> Seq.map (fun match' ->
        let resultVar = match'.Groups.[1].Value
        let funcName = match'.Groups.[2].Value
        let returnType = match'.Groups.[3].Value
        
        // Convert function calls to appropriate LLVM calls
        match funcName with
        | "printf" -> sprintf "  %%%s = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.empty, i64 0, i64 0))" resultVar
        | "printfn" -> sprintf "  %%%s = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.empty, i64 0, i64 0))" resultVar
        | "readLine" -> sprintf "  %%%s = call i32 @getchar()" resultVar
        | _ -> sprintf "  %%%s = call %s @%s()" resultVar returnType funcName
    )
    |> String.concat "\n"

/// Converts MLIR return statements to LLVM IR returns
let private convertReturnStatements (mlirText: string) (constants: Map<string, string>) : string =
    let returnPattern = @"func\.return\s*([^:\n]*)"
    let returnMatch = Regex.Match(mlirText, returnPattern)
    
    if returnMatch.Success then
        let returnValue = returnMatch.Groups.[1].Value.Trim()
        if String.IsNullOrEmpty(returnValue) then
            "  ret void"
        else
            let varName = returnValue.TrimStart('%')
            if constants.ContainsKey(varName) then
                sprintf "  ret i32 %%%s" varName
            else
                "  ret i32 0"
    else
        "  ret i32 0"

/// Translates a single MLIR function to LLVM IR
let private translateFunction (funcName: string) (returnType: string) (funcBody: string) : string =
    let functionSig = convertFunctionSignature funcName returnType
    let constants = convertConstants funcBody
    let functionCalls = convertFunctionCalls funcBody
    
    // Build simple constant map for return value resolution
    let constantMap = 
        let constantPattern = @"%(\w+)\s*=\s*arith\.constant\s+(\d+)\s*:\s*(\w+)"
        Regex.Matches(funcBody, constantPattern)
        |> Seq.cast<Match>
        |> Seq.fold (fun acc match' ->
            let varName = match'.Groups.[1].Value
            let value = match'.Groups.[2].Value
            Map.add varName value acc
        ) Map.empty
    
    let returnStmt = convertReturnStatements funcBody constantMap
    
    let bodyContent = [constants; functionCalls] |> List.filter (not << String.IsNullOrEmpty) |> String.concat "\n"
    
    sprintf "%s\nentry:\n%s\n%s\n}" functionSig bodyContent returnStmt

/// Gets the target triple for LLVM based on platform
let private getTargetTriple (target: string) : string =
    match target.ToLowerInvariant() with
    | "x86_64-pc-windows-msvc" -> "x86_64-w64-windows-gnu"
    | "x86_64-pc-linux-gnu" -> "x86_64-pc-linux-gnu"
    | "x86_64-apple-darwin" -> "x86_64-apple-darwin"
    | "embedded" -> "thumbv7em-none-eabihf"
    | _ -> 
        if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
            "x86_64-w64-windows-gnu"
        elif RuntimeInformation.IsOSPlatform(OSPlatform.Linux) then
            "x86_64-pc-linux-gnu"
        elif RuntimeInformation.IsOSPlatform(OSPlatform.OSX) then
            "x86_64-apple-darwin"
        else
            "x86_64-w64-windows-gnu"

/// Generates external function declarations needed for LLVM IR
let private generateExternalDeclarations (mlirText: string) : string =
    let declarations = ResizeArray<string>()
    
    if mlirText.Contains("printf") then
        declarations.Add("declare i32 @printf(i8*, ...)")
    
    if mlirText.Contains("printfn") then
        declarations.Add("declare i32 @printf(i8*, ...)")
    
    if mlirText.Contains("readLine") then
        declarations.Add("declare i32 @getchar()")
    
    declarations.Add("@.empty = private unnamed_addr constant [1 x i8] zeroinitializer, align 1")
    
    if declarations.Count > 0 then
        String.concat "\n" declarations + "\n\n"
    else
        ""

/// Translates MLIR (in LLVM dialect) to LLVM IR text
let translateToLLVM (mlirText: string) : LLVMOutput =
    let moduleName = extractModuleName mlirText
    
    // Parse functions from MLIR
    let funcPattern = @"func\.func\s+@(\w+)\s*\([^)]*\)\s*->\s*([^{]+)\s*\{([^}]+)\}"
    let matches = Regex.Matches(mlirText, funcPattern, RegexOptions.Singleline)
    
    let functions = 
        matches
        |> Seq.cast<Match>
        |> Seq.map (fun match' ->
            let funcName = match'.Groups.[1].Value
            let returnType = match'.Groups.[2].Value.Trim()
            let funcBody = match'.Groups.[3].Value
            translateFunction funcName returnType funcBody
        )
        |> String.concat "\n\n"
    
    let moduleHeader = sprintf "; ModuleID = '%s'\ntarget triple = \"x86_64-w64-windows-gnu\"\n\n" moduleName
    let externalDecls = generateExternalDeclarations mlirText
    let llvmIR = moduleHeader + externalDecls + functions
    
    // Build symbol table
    let symbolTable = 
        matches
        |> Seq.cast<Match>
        |> Seq.fold (fun acc match' ->
            let funcName = match'.Groups.[1].Value
            let returnType = match'.Groups.[2].Value.Trim()
            let signature = sprintf "%s ()" (if returnType = "()" then "void" else returnType)
            Map.add funcName signature acc
        ) Map.empty
    
    { 
        ModuleName = moduleName
        LLVMIRText = llvmIR
        SymbolTable = symbolTable
    }

/// Gets LLVM version information
let private getLLVMVersion() : string option =
    try
        let processInfo = System.Diagnostics.ProcessStartInfo()
        processInfo.FileName <- "llc"
        processInfo.Arguments <- "--version"
        processInfo.UseShellExecute <- false
        processInfo.RedirectStandardOutput <- true
        processInfo.RedirectStandardError <- true
        processInfo.CreateNoWindow <- true
        
        use proc = System.Diagnostics.Process.Start(processInfo)
        proc.WaitForExit(5000) |> ignore
        
        if proc.ExitCode = 0 then
            let output = proc.StandardOutput.ReadToEnd()
            let lines = output.Split('\n')
            let versionLine = lines |> Array.tryFind (fun line -> line.Contains("LLVM version"))
            versionLine
        else
            None
    with
    | _ -> None

/// Checks if a command is available in PATH
let private isCommandAvailable (command: string) : bool =
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

/// Gets the appropriate compiler commands for the current platform
let private getCompilerCommands (target: string) : (string * string) option =
    if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
        if isCommandAvailable "llc" && isCommandAvailable "gcc" then
            Some ("llc", "gcc")
        elif isCommandAvailable "llc" && isCommandAvailable "clang" then
            Some ("llc", "clang")
        elif isCommandAvailable "cl" then
            Some ("llc", "cl")
        else
            None
    else
        if isCommandAvailable "llc" && isCommandAvailable "clang" then
            Some ("llc", "clang")
        elif isCommandAvailable "gcc" then
            Some ("llc", "gcc")
        else
            None

/// Writes LLVM IR to file and compiles to native code
let compileLLVMToNative (llvmOutput: LLVMOutput) (outputPath: string) (target: string) : bool =
    try
        match getLLVMVersion() with
        | Some version -> printfn "Using %s" version
        | None -> printfn "Could not detect LLVM version"
        
        let llvmPath = Path.ChangeExtension(outputPath, ".ll")
        let utf8WithoutBom = System.Text.UTF8Encoding(false)
        File.WriteAllText(llvmPath, llvmOutput.LLVMIRText, utf8WithoutBom)
        printfn "Saved LLVM IR to: %s" llvmPath
        
        match getCompilerCommands target with
        | None ->
            printfn "Error: No suitable compiler found for target '%s'" target
            false
        
        | Some (llcCommand, linkerCommand) ->
            let objPath = Path.ChangeExtension(outputPath, ".o")
            let targetTriple = getTargetTriple target
            
            let llcInfo = System.Diagnostics.ProcessStartInfo()
            llcInfo.FileName <- llcCommand
            llcInfo.Arguments <- sprintf "-filetype=obj -mtriple=%s %s -o %s" targetTriple llvmPath objPath
            llcInfo.UseShellExecute <- false
            llcInfo.RedirectStandardOutput <- true
            llcInfo.RedirectStandardError <- true
            
            use llcProc = System.Diagnostics.Process.Start(llcInfo)
            llcProc.WaitForExit()
            
            if llcProc.ExitCode <> 0 then
                let error = llcProc.StandardError.ReadToEnd()
                printfn "LLC compilation failed: %s" error
                false
            else
                let linkInfo = System.Diagnostics.ProcessStartInfo()
                linkInfo.FileName <- linkerCommand
                linkInfo.Arguments <- sprintf "%s -o %s" objPath outputPath
                linkInfo.UseShellExecute <- false
                linkInfo.RedirectStandardOutput <- true
                linkInfo.RedirectStandardError <- true
                
                use linkProc = System.Diagnostics.Process.Start(linkInfo)
                linkProc.WaitForExit()
                
                if linkProc.ExitCode <> 0 then
                    let error = linkProc.StandardError.ReadToEnd()
                    printfn "Linking failed: %s" error
                    false
                else
                    if File.Exists(llvmPath) && not (llvmPath.EndsWith(outputPath + ".ll")) then 
                        File.Delete(llvmPath)
                    if File.Exists(objPath) then File.Delete(objPath)
                    true
    with
    | ex ->
        printfn "Error during compilation: %s" ex.Message
        false