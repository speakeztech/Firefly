module Core.Conversion.LLVMTranslator

open System
open System.IO
open System.Text.RegularExpressions
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
    
    sprintf "%s\nentry:\n%s\n%s\n}" functionSig constants returnStmt

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
    
    let moduleHeader = sprintf "; ModuleID = '%s'\ntarget triple = \"x86_64-pc-linux-gnu\"\n\n" moduleName
    let llvmIR = moduleHeader + functions
    
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

/// Writes LLVM IR to file and compiles to native code
let compileLLVMToNative (llvmOutput: LLVMOutput) (outputPath: string) : bool =
    try
        // Write LLVM IR to temporary file
        let llvmPath = Path.ChangeExtension(outputPath, ".ll")
        File.WriteAllText(llvmPath, llvmOutput.LLVMIRText)
        
        // Compile LLVM IR to object file using llc
        let objPath = Path.ChangeExtension(outputPath, ".o")
        let llcInfo = System.Diagnostics.ProcessStartInfo()
        llcInfo.FileName <- "llc"
        llcInfo.Arguments <- sprintf "-filetype=obj %s -o %s" llvmPath objPath
        llcInfo.UseShellExecute <- false
        llcInfo.RedirectStandardOutput <- true
        llcInfo.RedirectStandardError <- true
        
        use llcProcess = System.Diagnostics.Process.Start(llcInfo)
        llcProcess.WaitForExit()
        
        if llcProcess.ExitCode <> 0 then
            printfn "LLC compilation failed"
            false
        else
            // Link object file to create executable
            let linkInfo = System.Diagnostics.ProcessStartInfo()
            linkInfo.FileName <- "clang"
            linkInfo.Arguments <- sprintf "%s -o %s" objPath outputPath
            linkInfo.UseShellExecute <- false
            linkInfo.RedirectStandardOutput <- true
            linkInfo.RedirectStandardError <- true
            
            use linkProcess = System.Diagnostics.Process.Start(linkInfo)
            linkProcess.WaitForExit()
            
            // Clean up intermediate files
            if File.Exists(llvmPath) then File.Delete(llvmPath)
            if File.Exists(objPath) then File.Delete(objPath)
            
            linkProcess.ExitCode = 0
    with
    | ex ->
        printfn "Error during compilation: %s" ex.Message
        false