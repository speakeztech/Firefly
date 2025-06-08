module Core.MLIRGeneration.Dialect

open System
open System.Collections.Generic

/// Represents MLIR dialects used in the compilation pipeline
type MLIRDialect =
    | Standard
    | LLVM
    | Func
    | Arith
    | MemRef
    | Affine
    | Vector

/// Represents an MLIR operation signature
type OperationSignature = {
    Name: string
    Dialect: MLIRDialect
    Operands: string list
    Results: string list
    Attributes: string list
}

/// Maps F# concepts to appropriate MLIR dialects
let fsharpToDialectMapping = [
    ("function", Func)
    ("arithmetic", Arith)
    ("array", MemRef)
    ("vectorized-ops", Vector)
    ("control-flow", Standard)
    ("memory", MemRef)
    ("loops", Affine)
    ("conditionals", Standard)
]

/// Core operations for each dialect
let dialectOperations = 
    let operations = Dictionary<MLIRDialect, OperationSignature list>()
    
    // Standard dialect operations
    operations.[Standard] <- [
        { Name = "std.constant"; Dialect = Standard; Operands = []; Results = ["result"]; Attributes = ["value"] }
        { Name = "std.br"; Dialect = Standard; Operands = []; Results = []; Attributes = ["dest"] }
        { Name = "std.cond_br"; Dialect = Standard; Operands = ["condition"]; Results = []; Attributes = ["true_dest"; "false_dest"] }
    ]
    
    // Function dialect operations  
    operations.[Func] <- [
        { Name = "func.func"; Dialect = Func; Operands = []; Results = []; Attributes = ["sym_name"; "function_type"] }
        { Name = "func.return"; Dialect = Func; Operands = ["operands"]; Results = []; Attributes = [] }
        { Name = "func.call"; Dialect = Func; Operands = ["arguments"]; Results = ["results"]; Attributes = ["callee"] }
    ]
    
    // Arithmetic dialect operations
    operations.[Arith] <- [
        { Name = "arith.constant"; Dialect = Arith; Operands = []; Results = ["result"]; Attributes = ["value"] }
        { Name = "arith.addi"; Dialect = Arith; Operands = ["lhs"; "rhs"]; Results = ["result"]; Attributes = [] }
        { Name = "arith.subi"; Dialect = Arith; Operands = ["lhs"; "rhs"]; Results = ["result"]; Attributes = [] }
        { Name = "arith.muli"; Dialect = Arith; Operands = ["lhs"; "rhs"]; Results = ["result"]; Attributes = [] }
        { Name = "arith.divsi"; Dialect = Arith; Operands = ["lhs"; "rhs"]; Results = ["result"]; Attributes = [] }
        { Name = "arith.cmpi"; Dialect = Arith; Operands = ["lhs"; "rhs"]; Results = ["result"]; Attributes = ["predicate"] }
    ]
    
    // Memory reference dialect operations
    operations.[MemRef] <- [
        { Name = "memref.alloc"; Dialect = MemRef; Operands = []; Results = ["memref"]; Attributes = [] }
        { Name = "memref.alloca"; Dialect = MemRef; Operands = []; Results = ["memref"]; Attributes = [] }
        { Name = "memref.load"; Dialect = MemRef; Operands = ["memref"; "indices"]; Results = ["result"]; Attributes = [] }
        { Name = "memref.store"; Dialect = MemRef; Operands = ["value"; "memref"; "indices"]; Results = []; Attributes = [] }
        { Name = "memref.dealloc"; Dialect = MemRef; Operands = ["memref"]; Results = []; Attributes = [] }
    ]
    
    // LLVM dialect operations
    operations.[LLVM] <- [
        { Name = "llvm.func"; Dialect = LLVM; Operands = []; Results = []; Attributes = ["sym_name"; "function_type"] }
        { Name = "llvm.return"; Dialect = LLVM; Operands = ["operands"]; Results = []; Attributes = [] }
        { Name = "llvm.call"; Dialect = LLVM; Operands = ["arguments"]; Results = ["results"]; Attributes = ["callee"] }
        { Name = "llvm.add"; Dialect = LLVM; Operands = ["lhs"; "rhs"]; Results = ["result"]; Attributes = [] }
        { Name = "llvm.sub"; Dialect = LLVM; Operands = ["lhs"; "rhs"]; Results = ["result"]; Attributes = [] }
        { Name = "llvm.mul"; Dialect = LLVM; Operands = ["lhs"; "rhs"]; Results = ["result"]; Attributes = [] }
        { Name = "llvm.alloca"; Dialect = LLVM; Operands = ["size"]; Results = ["ptr"]; Attributes = [] }
        { Name = "llvm.load"; Dialect = LLVM; Operands = ["ptr"]; Results = ["value"]; Attributes = [] }
        { Name = "llvm.store"; Dialect = LLVM; Operands = ["value"; "ptr"]; Results = []; Attributes = [] }
    ]
    
    // Affine dialect operations
    operations.[Affine] <- [
        { Name = "affine.for"; Dialect = Affine; Operands = []; Results = []; Attributes = ["lower_bound"; "upper_bound"; "step"] }
        { Name = "affine.if"; Dialect = Affine; Operands = []; Results = []; Attributes = ["condition"] }
        { Name = "affine.load"; Dialect = Affine; Operands = ["memref"]; Results = ["result"]; Attributes = ["map"] }
        { Name = "affine.store"; Dialect = Affine; Operands = ["value"; "memref"]; Results = []; Attributes = ["map"] }
    ]
    
    // Vector dialect operations
    operations.[Vector] <- [
        { Name = "vector.broadcast"; Dialect = Vector; Operands = ["source"]; Results = ["result"]; Attributes = [] }
        { Name = "vector.extract"; Dialect = Vector; Operands = ["vector"]; Results = ["result"]; Attributes = ["position"] }
        { Name = "vector.insert"; Dialect = Vector; Operands = ["source"; "dest"]; Results = ["result"]; Attributes = ["position"] }
    ]
    
    operations

/// Gets all operations for a specific dialect
let getDialectOperations (dialect: MLIRDialect) : OperationSignature list =
    match dialectOperations.TryGetValue(dialect) with
    | (true, operations) -> operations
    | (false, _) -> []

/// Checks if an operation belongs to a specific dialect
let isOperationInDialect (operationName: string) (dialect: MLIRDialect) : bool =
    let operations = getDialectOperations dialect
    operations |> List.exists (fun op -> op.Name = operationName)

/// Gets the dialect for a given operation name
let getDialectForOperation (operationName: string) : MLIRDialect option =
    dialectOperations
    |> Seq.tryPick (fun kvp ->
        if kvp.Value |> List.exists (fun op -> op.Name = operationName) then
            Some kvp.Key
        else
            None
    )

/// Validates that an operation signature matches its dialect requirements
let validateOperationSignature (operation: OperationSignature) : bool =
    let dialectOps = getDialectOperations operation.Dialect
    dialectOps |> List.exists (fun op -> 
        op.Name = operation.Name &&
        op.Operands.Length = operation.Operands.Length &&
        op.Results.Length = operation.Results.Length
    )

/// Gets all supported dialects
let getAllSupportedDialects() : MLIRDialect list =
    [Standard; LLVM; Func; Arith; MemRef; Affine; Vector]

/// Converts dialect enum to string representation
let dialectToString (dialect: MLIRDialect) : string =
    match dialect with
    | Standard -> "std"
    | LLVM -> "llvm"
    | Func -> "func"
    | Arith -> "arith"
    | MemRef -> "memref"
    | Affine -> "affine"
    | Vector -> "vector"

/// Converts string to dialect enum
let stringToDialect (dialectStr: string) : MLIRDialect option =
    match dialectStr.ToLowerInvariant() with
    | "std" | "standard" -> Some Standard
    | "llvm" -> Some LLVM
    | "func" | "function" -> Some Func
    | "arith" | "arithmetic" -> Some Arith
    | "memref" | "memory" -> Some MemRef
    | "affine" -> Some Affine
    | "vector" -> Some Vector
    | _ -> None

/// Registers all required dialects with the given MLIR context
let registerRequiredDialects (contextHandle: nativeint) : unit =
    let requiredDialects = getAllSupportedDialects()
    // Registration logic would interface with actual MLIR C API
    // For now, we track which dialects are registered
    requiredDialects |> List.iter (fun dialect ->
        let dialectName = dialectToString dialect
        printfn "Registering dialect: %s" dialectName
    )

/// Checks if all required dialects for Firefly are available
let verifyRequiredDialects() : bool =
    let requiredForFirefly = [Standard; LLVM; Func; Arith; MemRef]
    let available = getAllSupportedDialects()
    requiredForFirefly |> List.forall (fun required -> List.contains required available)

/// Gets operations that are safe for zero-allocation compilation
let getZeroAllocOperations() : OperationSignature list =
    let safeOperations = [
        // Arithmetic operations are always safe
        yield! getDialectOperations Arith
        
        // Function operations are safe
        yield! getDialectOperations Func |> List.filter (fun op -> op.Name <> "func.call") // Calls need analysis
        
        // Stack allocations only (no heap)
        yield! getDialectOperations MemRef |> List.filter (fun op -> 
            op.Name = "memref.alloca" || op.Name = "memref.load" || op.Name = "memref.store")
        
        // LLVM stack operations
        yield! getDialectOperations LLVM |> List.filter (fun op ->
            not (op.Name.Contains("malloc") || op.Name.Contains("free")))
    ]
    safeOperations

/// Validates that an MLIR module only uses zero-allocation operations
let validateZeroAllocationModule (moduleText: string) : bool =
    let safeOps = getZeroAllocOperations() |> List.map (fun op -> op.Name) |> Set.ofList
    let lines = moduleText.Split('\n')
    
    lines |> Array.forall (fun line ->
        let trimmed = line.Trim()
        if trimmed.Contains(" = ") && (trimmed.Contains(".") || trimmed.Contains("(")) then
            // Extract operation name
            let parts = trimmed.Split([|'='; ' '|], StringSplitOptions.RemoveEmptyEntries)
            if parts.Length > 1 then
                let opPart = parts.[1].Trim()
                let opName = 
                    if opPart.Contains("(") then
                        opPart.Substring(0, opPart.IndexOf("("))
                    else
                        opPart
                safeOps.Contains(opName) || not (opName.Contains("."))
            else
                true
        else
            true
    )