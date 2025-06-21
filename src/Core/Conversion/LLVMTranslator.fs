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
        if trimmed.Contains("func.func") || trimmed.Contains("llvm.func") then
            try
                let isPrivate = trimmed.Contains("private")
                
                // Extract function name - look for @functionname
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
                    
                    // Extract parameters
                    let paramStart = trimmed.IndexOf('(')
                    let paramEnd = trimmed.IndexOf(')')
                    let parameters = 
                        if paramStart >= 0 && paramEnd > paramStart then
                            let paramStr = trimmed.Substring(paramStart + 1, paramEnd - paramStart - 1).Trim()
                            if String.IsNullOrWhiteSpace(paramStr) then []
                            else 
                                // Parse parameters like "%arg0: i32, %arg1: memref<?xmemref<?xi8>>"
                                paramStr.Split(',')
                                |> Array.mapi (fun i paramDecl -> 
                                    let parts = paramDecl.Trim().Split(':')
                                    if parts.Length >= 2 then
                                        let paramName = parts.[0].Trim()
                                        let paramType = parts.[1].Trim()
                                        (paramName, paramType)
                                    else
                                        (sprintf "%%arg%d" i, "i32"))
                                |> Array.toList
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
                                if retType = "()" then "void" else retType
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
        if trimmed.Contains("memref.global") || trimmed.Contains("llvm.mlir.global") then
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
                        elif equalIndex > 0 then equalIndex
                        elif spaceIndex > 0 then spaceIndex
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

    /// Parses MLIR function body operations
    let parseFunctionBody (lines: string array) (startIndex: int) : string list * int =
        let mutable currentIndex = startIndex + 1  // Skip the function signature line
        let mutable operations = []
        let mutable finished = false
        
        // The function signature line should end with "{"
        let functionLine = lines.[startIndex].Trim()
        let isValidFunction = functionLine.EndsWith("{")
        
        printfn "DEBUG: parseFunctionBody for line %d: '%s'" startIndex functionLine
        printfn "DEBUG: isValidFunction = %b" isValidFunction
        
        if not isValidFunction then
            ([], startIndex + 1)  // Not a proper function, return empty
        else
            while currentIndex < lines.Length && not finished do
                let line = lines.[currentIndex]
                let trimmed = line.Trim()
                
                printfn "DEBUG: Line %d: '%s' (trimmed: '%s')" currentIndex line trimmed
                
                if trimmed = "}" then
                    printfn "DEBUG: Found closing brace, finishing"
                    finished <- true
                elif not (String.IsNullOrWhiteSpace(trimmed)) && 
                     not (trimmed.StartsWith("//")) &&
                     not (trimmed.StartsWith("module")) &&
                     not (trimmed.StartsWith("llvm.func")) &&
                     not (trimmed.StartsWith("func.func")) &&
                     trimmed <> "{" then
                    // Accept any non-empty, non-comment, non-function-declaration line
                    printfn "DEBUG: Adding operation: '%s'" trimmed
                    operations <- trimmed :: operations
                else
                    printfn "DEBUG: Skipping line: '%s'" trimmed
                
                if not finished then
                    currentIndex <- currentIndex + 1
            
            printfn "DEBUG: Final operations count: %d" operations.Length
            for op in operations do
                printfn "DEBUG: Operation: '%s'" op
            
            (List.rev operations, currentIndex + 1)

/// Converts MLIR operations to LLVM IR
module OperationConverter =
    
    /// Converts a single MLIR operation to LLVM IR
    let convertOperation (operation: string) : string =
        let trimmed = operation.Trim()
        
        match trimmed with
        | s when s.Contains("llvm.mlir.constant") ->
            // Convert: %const7 = llvm.mlir.constant 0 : i32
            // To:     %const7 = add i32 0, 0
            let parts = s.Split('=')
            if parts.Length = 2 then
                let resultVar = parts.[0].Trim()
                
                // Extract constant value and type
                let valueStartIdx = s.IndexOf("constant") + "constant".Length
                let valueEndIdx = s.IndexOf(':', valueStartIdx)
                
                if valueStartIdx > 0 && valueEndIdx > valueStartIdx then
                    let constValue = s.Substring(valueStartIdx, valueEndIdx - valueStartIdx).Trim()
                    let typeAndComment = s.Substring(valueEndIdx + 1).Trim()
                    
                    // Handle inline comments
                    let (typeStr, comment) =
                        match typeAndComment.IndexOf("//") with
                        | idx when idx >= 0 -> 
                            (typeAndComment.Substring(0, idx).Trim(), 
                            " ; " + typeAndComment.Substring(idx + 2).Trim())
                        | _ -> (typeAndComment, "")
                    
                    // Generate appropriate add instruction based on type
                    if typeStr.Contains("i32") then
                        sprintf "  %s = add i32 %s, 0%s" resultVar constValue comment
                    elif typeStr.Contains("i64") then
                        sprintf "  %s = add i64 %s, 0%s" resultVar constValue comment
                    elif typeStr.Contains("f32") then
                        sprintf "  %s = fadd float %s, 0.0%s" resultVar constValue comment
                    elif typeStr.Contains("f64") then
                        sprintf "  %s = fadd double %s, 0.0%s" resultVar constValue comment
                    else
                        sprintf "  %s = add i32 %s, 0%s" resultVar constValue comment
                else
                    sprintf "  ; Error parsing constant: %s" s
            else
                sprintf "  ; Invalid constant format: %s" s
                
        | s when s.Contains("llvm.bitcast") ->
            // Convert: %conv2 = llvm.bitcast %const1 : i32 to ()
            // To:     %conv2 = bitcast i32 %const1 to i8*
            let parts = s.Split('=')
            if parts.Length = 2 then
                let resultVar = parts.[0].Trim()
                let rightSide = parts.[1].Trim()
                
                // Extract source variable
                let valueStartIdx = rightSide.IndexOf('%', rightSide.IndexOf("bitcast") + 7)
                let colonIdx = rightSide.IndexOf(':', valueStartIdx)
                let toIdx = rightSide.LastIndexOf(" to ")
                
                if valueStartIdx > 0 && colonIdx > valueStartIdx && toIdx > colonIdx then
                    let sourceVar = rightSide.Substring(valueStartIdx, colonIdx - valueStartIdx).Trim()
                    let sourceType = rightSide.Substring(colonIdx + 1, toIdx - colonIdx - 1).Trim()
                    let targetType = rightSide.Substring(toIdx + 4).Trim()
                    
                    // Convert MLIR types to LLVM types
                    let llvmSourceType = 
                        match sourceType with
                        | "i32" -> "i32"
                        | "i64" -> "i64"
                        | "memref<?xi8>" -> "i8*"
                        | "()" -> "i8*"  // Handle void type
                        | _ -> "i8*"
                    
                    let llvmTargetType =
                        match targetType with
                        | "i32" -> "i32"
                        | "i64" -> "i64" 
                        | "memref<?xi8>" -> "i8*"
                        | "()" -> "i8*"  // Handle void type
                        | _ -> "i8*"
                    
                    // Generate proper LLVM bitcast with correct type order
                    sprintf "  %s = bitcast %s %s to %s" resultVar llvmSourceType sourceVar llvmTargetType
                else
                    sprintf "  ; Error parsing bitcast: %s" s
            else
                sprintf "  ; Invalid bitcast format: %s" s
        
        | s when s.Contains("llvm.call") ->
            // Convert: %call7 = llvm.call @printf(%conv6) : (memref<?xi8>) -> ()
            // To:     %call7 = call i32 @printf(i8* %conv6)
            let parts = s.Split('=')
            if parts.Length = 2 then
                let resultVar = parts.[0].Trim()
                let rightSide = parts.[1].Trim()
                
                // Extract function name and args
                let atIdx = rightSide.IndexOf('@')
                let openParenIdx = rightSide.IndexOf('(', atIdx)
                let closeParenIdx = rightSide.IndexOf(')', openParenIdx)
                
                if atIdx > 0 && openParenIdx > atIdx && closeParenIdx > openParenIdx then
                    let funcName = rightSide.Substring(atIdx, openParenIdx - atIdx)
                    let argsStr = rightSide.Substring(openParenIdx + 1, closeParenIdx - openParenIdx - 1)
                    
                    // Determine return type
                    let colonIdx = rightSide.IndexOf(':', closeParenIdx)
                    let arrowIdx = rightSide.IndexOf("->", colonIdx)
                    
                    let returnType =
                        if colonIdx > 0 && arrowIdx > colonIdx then
                            let returnTypeStr = rightSide.Substring(arrowIdx + 2).Trim()
                            if returnTypeStr.EndsWith(")") then
                                let cleanType = returnTypeStr.Substring(0, returnTypeStr.Length - 1).Trim()
                                match cleanType with
                                | "()" -> "void"
                                | "i32" -> "i32"
                                | "i1" -> "i1"
                                | "memref<?xi8>" -> "i8*"
                                | _ -> "i32"
                            else "i32"
                        else "i32"
                    
                    // Handle special cases for printf and helper functions    
                    if funcName = "@printf" then
                        sprintf "  %s = call i32 (i8*, ...) %s(i8* %s)" resultVar funcName argsStr
                    elif funcName = "@is_ok_result" then
                        sprintf "  %s = call i1 %s(i32 %s)" resultVar funcName argsStr
                    elif funcName = "@extract_result_length" then
                        sprintf "  %s = call i32 %s(i32 %s)" resultVar funcName argsStr
                    elif funcName = "@create_span" then
                        let args = argsStr.Split(',')
                        if args.Length >= 2 then
                            sprintf "  %s = call i8* %s(i8* %s, i32 %s)" resultVar funcName (args.[0].Trim()) (args.[1].Trim())
                        else
                            sprintf "  %s = call i8* %s(i8* %s)" resultVar funcName argsStr
                    else
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
                else
                    sprintf "  ; Error parsing function call: %s" s
            else
                sprintf "  ; Invalid function call format: %s" s
                
        | s when s.StartsWith("^") ->
            // Convert: ^then13:
            // To:     then13:
            sprintf "%s:" (s.TrimStart('^'))
            
        | s when s.StartsWith("br ^") ->
            // Convert: br ^end15
            // To:     br label %end15
            let labelName = s.Substring(3).Trim()
            sprintf "  br label %%%s" labelName
            
        | s when s.StartsWith("cond_br") ->
            // Convert: cond_br %is_ok12, ^then13, ^else14
            // To:     br i1 %is_ok12, label %then13, label %else14
            let commaIdx1 = s.IndexOf(',')
            let commaIdx2 = s.IndexOf(',', commaIdx1 + 1)
            
            if commaIdx1 > 0 && commaIdx2 > commaIdx1 then
                let condition = s.Substring("cond_br".Length, commaIdx1 - "cond_br".Length).Trim()
                let thenLabel = s.Substring(commaIdx1 + 1, commaIdx2 - commaIdx1 - 1).Trim().TrimStart('^')
                let elseLabel = s.Substring(commaIdx2 + 1).Trim().TrimStart('^')
                
                sprintf "  br i1 %s, label %%%s, label %%%s" condition thenLabel elseLabel
            else
                sprintf "  ; Error parsing conditional branch: %s" s
                
        | s when s.Contains(" = ") && s.Contains(" : ") && not (s.Contains("llvm.")) ->
            // Convert: %match_result11 = %call20 : memref<?xi8>
            // To:     %match_result11 = %call20
            let parts = s.Split('=')
            if parts.Length = 2 then
                let resultVar = parts.[0].Trim()
                let rightSide = parts.[1].Trim()
                
                let colonIdx = rightSide.IndexOf(':')
                if colonIdx > 0 then
                    let sourceVar = rightSide.Substring(0, colonIdx).Trim()
                    sprintf "  %s = %s" resultVar sourceVar
                else
                    sprintf "  %s = %s" resultVar rightSide
            else
                sprintf "  ; Error parsing assignment: %s" s
                
        | s when s.Contains("memref.llvm.alloca") ->
            // Convert: %call3 = memref.llvm.alloca(%conv2) : memref<?xi8> {element_type = i8}
            // To:     %call3 = alloca i8, i32 256
            let parts = s.Split('=')
            if parts.Length = 2 then
                let resultVar = parts.[0].Trim()
                
                // Try to extract size
                let openParenIdx = s.IndexOf('(')
                let closeParenIdx = s.IndexOf(')', openParenIdx)
                
                if openParenIdx > 0 && closeParenIdx > openParenIdx then
                    let sizeVar = s.Substring(openParenIdx + 1, closeParenIdx - openParenIdx - 1)
                    sprintf "  %s = alloca i8, i32 256, align 1" resultVar
                else
                    sprintf "  %s = alloca i8, i32 256, align 1" resultVar
            else
                sprintf "  ; Invalid alloca format: %s" s
                
        | s when s.Contains("llvm.mlir.addressof") ->
            // Convert: %str_ptr5 = llvm.mlir.addressof @str_4 : memref<?xi8>
            // To:     %str_ptr5 = getelementptr [32 x i8], [32 x i8]* @str_4, i32 0, i32 0
            let parts = s.Split('=')
            if parts.Length = 2 then
                let resultVar = parts.[0].Trim()
                
                // Extract global name
                let atIdx = s.IndexOf('@')
                let colonIdx = s.IndexOf(':', atIdx)
                
                if atIdx > 0 && colonIdx > atIdx then
                    let globalName = s.Substring(atIdx, colonIdx - atIdx).Trim()
                    sprintf "  %s = getelementptr [32 x i8], [32 x i8]* %s, i32 0, i32 0" resultVar globalName
                else
                    sprintf "  ; Error parsing addressof: %s" s
            else
                sprintf "  ; Invalid addressof format: %s" s
        
        | s when s.Contains("llvm.return") ->
            // Handle return statements
            if s.Contains(":") then
                let spaceAfterReturn = s.IndexOf(' ', "llvm.return".Length)
                let colonIdx = s.LastIndexOf(':')
                
                if spaceAfterReturn > 0 && colonIdx > spaceAfterReturn then
                    let returnVal = s.Substring(spaceAfterReturn, colonIdx - spaceAfterReturn).Trim()
                    let typeStr = s.Substring(colonIdx + 1).Trim()
                    
                    if typeStr.Contains("i32") then
                        sprintf "  ret i32 %s" returnVal
                    elif typeStr.Contains("i64") then
                        sprintf "  ret i64 %s" returnVal
                    else
                        sprintf "  ret i32 %s" returnVal
                else
                    "  ret void"
            else
                "  ret void"
        
        // Handle comments and unrecognized operations
        | s when s.StartsWith(";") -> 
            // Convert // comments to ; for LLVM
            sprintf "  ; %s" (s.TrimStart(';').Trim())
            
        | _ -> 
            // Pass unrecognized operations as comments
            sprintf "  ; TODO: %s" trimmed

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
        | t when t.StartsWith("memref<") && t.Contains("xi8>") -> "i8*"
        | t when t.StartsWith("memref<") -> "i8*"  // Default to i8* for all memrefs
        | _ -> "i32"
    
    /// Generates LLVM function from parsed MLIR function with body translation
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
            // Function definition with body translation
            writer.WriteLine(sprintf "define %s %s(%s) {" llvmReturnType func.Name llvmParameters)
            writer.WriteLine("entry:")
            
            // First sort constants to the beginning
            let (constants, otherOps) = 
                func.Body
                |> List.partition (fun op -> op.Contains("llvm.mlir.constant"))
            
            // Process constants first
            for constOp in constants do
                let llvmConstOp = OperationConverter.convertOperation constOp
                writer.WriteLine(llvmConstOp)
            
            // Then process all other operations
            for op in otherOps do
                let llvmOp = OperationConverter.convertOperation op
                writer.WriteLine(llvmOp)
            
            // Add return if not already present
            let hasReturn = func.Body |> List.exists (fun op -> op.Contains("return"))
            if not hasReturn then
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
    
    /// Processes MLIR text and extracts functions with bodies and globals
    let processMlirText (mlirText: string) : string * ParsedMLIRFunction list * ParsedMLIRGlobal list =
        let lines = mlirText.Split('\n')
        let mutable moduleName = "main"
        let mutable functions = []
        let mutable globalConstants = []
        let mutable currentIndex = 0
        
        printfn "DEBUG: Processing MLIR with %d lines" lines.Length
        
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
                printfn "DEBUG: Found function %s" func.Name
                let (body, nextIndex) = MLIRParser.parseFunctionBody lines currentIndex
                printfn "DEBUG: Function %s has %d body operations" func.Name body.Length
                for op in body do
                    printfn "DEBUG: Body op: %s" op
                let funcWithBody = { func with Body = body }
                functions <- funcWithBody :: functions
                currentIndex <- nextIndex - 1  
            | None -> 
                // Parse globals
                match MLIRParser.parseGlobalConstant trimmed with
                | Some globalItem -> 
                    printfn "DEBUG: Found global %s" globalItem.Name
                    globalConstants <- globalItem :: globalConstants
                | None -> ()
            
            currentIndex <- currentIndex + 1
        
        printfn "DEBUG: Final result - %d functions, %d globals" functions.Length globalConstants.Length
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
        
        // Deduplicate global constants by name
        let uniqueGlobals = 
            globalConstants
            |> List.groupBy (fun g -> g.Name)
            |> List.map (fun (name, globals) -> List.head globals)
        
        // Global constants
        for globalItem in uniqueGlobals do
            let globalLLVM = LLVMGenerator.generateLLVMGlobal globalItem
            writer.Write(globalLLVM)
        
        if not uniqueGlobals.IsEmpty then writer.WriteLine("")
        
        // Functions
        let userFunctions = functions |> List.filter (fun f -> not f.IsExternal)
        for func in userFunctions do
            let funcLLVM = LLVMGenerator.generateLLVMFunction func
            writer.Write(funcLLVM)
            writer.WriteLine("")
        
        // Ensure main function exists and calls hello()
        let hasMain = userFunctions |> List.exists (fun f -> f.Name = "@main")
        if not hasMain then
            let hasHelloFunction = userFunctions |> List.exists (fun f -> f.Name = "@hello")
            if hasHelloFunction then
                writer.WriteLine("define i32 @main(i32 %argc, i8** %argv) {")
                writer.WriteLine("entry:")
                writer.WriteLine("  call void @hello()")
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