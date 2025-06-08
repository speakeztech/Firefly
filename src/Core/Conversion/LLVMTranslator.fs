module Core.Conversion.LLVMTranslator

open System
open System.IO
open System.Text.RegularExpressions
open Llvm.NET
open Llvm.NET.Values
open Llvm.NET.Types
open Core.MLIRGeneration.Dialect

/// Represents LLVM IR output from the translation process
type LLVMOutput = {
    ModuleName: string
    Module: Module
    LLVMIRText: string
    SymbolTable: Map<string, GlobalValue>
}

/// Extracts module name from MLIR text
let private extractModuleName (mlirText: string) : string =
    let modulePattern = @"module\s+@?(\w+)"
    let match' = Regex.Match(mlirText, modulePattern)
    if match'.Success then
        match'.Groups.[1].Value
    else
        "main"

/// Parses MLIR function declarations and creates LLVM functions
let private createLLVMFunctions (context: Context) (module': Module) (mlirText: string) : Map<string, Function> =
    let funcPattern = @"func\.func\s+@(\w+)\s*\([^)]*\)\s*->\s*([^{]+)"
    let matches = Regex.Matches(mlirText, funcPattern)
    
    matches
    |> Seq.cast<Match>
    |> Seq.fold (fun acc match' ->
        let funcName = match'.Groups.[1].Value
        let returnTypeStr = match'.Groups.[2].Value.Trim()
        
        let returnType = 
            match returnTypeStr with
            | "i32" -> context.Int32Type :> ITypeRef
            | "i64" -> context.Int64Type :> ITypeRef
            | "f32" -> context.FloatType :> ITypeRef
            | "f64" -> context.DoubleType :> ITypeRef
            | "()" | "void" -> context.VoidType :> ITypeRef
            | _ -> context.Int32Type :> ITypeRef
        
        let funcType = context.GetFunctionType(returnType, [||])
        let func = module'.AddFunction(funcName, funcType)
        
        Map.add funcName func acc
    ) Map.empty

/// Translates MLIR basic blocks to LLVM basic blocks
let private translateBasicBlocks (context: Context) (builder: IRBuilder) (func: Function) (mlirText: string) =
    let entryBlock = func.AppendBasicBlock("entry")
    builder.PositionAtEnd(entryBlock)
    
    // Parse constants
    let constantPattern = @"%(\w+)\s*=\s*arith\.constant\s+(\d+)\s*:\s*(\w+)"
    let constants = 
        Regex.Matches(mlirText, constantPattern)
        |> Seq.cast<Match>
        |> Seq.fold (fun acc match' ->
            let varName = match'.Groups.[1].Value
            let value = Int32.Parse(match'.Groups.[2].Value)
            let llvmValue = context.CreateConstant(value)
            Map.add varName llvmValue acc
        ) Map.empty
    
    // Handle return statements
    let returnPattern = @"func\.return\s*([^:\n]*)"
    let returnMatch = Regex.Match(mlirText, returnPattern)
    
    if returnMatch.Success then
        let returnValue = returnMatch.Groups.[1].Value.Trim()
        if String.IsNullOrEmpty(returnValue) then
            builder.Return() |> ignore
        else
            match constants.TryFind(returnValue.TrimStart('%')) with
            | Some value -> builder.Return(value) |> ignore
            | None -> 
                let zeroValue = context.CreateConstant(0)
                builder.Return(zeroValue) |> ignore

/// Translates MLIR (in LLVM dialect) to LLVM IR
let translateToLLVM (mlirText: string) : LLVMOutput =
    use context = new Context()
    let moduleName = extractModuleName mlirText
    let module' = context.CreateModule(moduleName)
    
    let functions = createLLVMFunctions context module' mlirText
    let builder = new IRBuilder(context)
    
    // Process each function
    functions
    |> Map.iter (fun name func ->
        let funcPattern = sprintf @"func\.func\s+@%s[^}]+}" name
        let funcMatch = Regex.Match(mlirText, funcPattern, RegexOptions.Singleline)
        if funcMatch.Success then
            translateBasicBlocks context builder func funcMatch.Value
    )
    
    let llvmIR = module'.WriteToString()
    let symbolTable = 
        functions
        |> Map.map (fun _ func -> func :> GlobalValue)
    
    { 
        ModuleName = moduleName
        Module = module'
        LLVMIRText = llvmIR
        SymbolTable = symbolTable
    }

/// Compiles LLVM IR to native code using LLVM backend
let compileLLVMToNative (llvmOutput: LLVMOutput) (outputPath: string) : bool =
    try
        use targetMachine = Target.DefaultTarget.CreateTargetMachine(
            Triple.HostTriple,
            Target.DefaultTarget.HostCPU,
            "",
            CodeGenOpt.Default,
            Reloc.Default,
            CodeModel.Default
        )
        
        // Write object file
        let objPath = Path.ChangeExtension(outputPath, ".o")
        targetMachine.EmitToFile(llvmOutput.Module, objPath, CodeGenFileType.ObjectFile)
        
        // Link to create executable
        let linkerArgs = [|
            objPath
            "-o"
            outputPath
        |]
        
        let processInfo = System.Diagnostics.ProcessStartInfo()
        processInfo.FileName <- "clang"
        processInfo.Arguments <- String.Join(" ", linkerArgs)
        processInfo.UseShellExecute <- false
        processInfo.RedirectStandardOutput <- true
        processInfo.RedirectStandardError <- true
        
        use linker = System.Diagnostics.Process.Start(processInfo)
        linker.WaitForExit()
        
        // Clean up object file
        if File.Exists(objPath) then
            File.Delete(objPath)
        
        linker.ExitCode = 0
    with
    | ex ->
        printfn "Error during compilation: %s" ex.Message
        false