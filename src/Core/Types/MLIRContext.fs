module Core.Types.MLIRContext

open System
open System.Runtime.InteropServices
open Core.Types.TypeSystem

/// MLIR C API function imports
[<DllImport("libMLIR-C.so", CallingConvention = CallingConvention.Cdecl)>]
extern nativeint mlirContextCreate()

[<DllImport("libMLIR-C.so", CallingConvention = CallingConvention.Cdecl)>]
extern void mlirContextDestroy(nativeint context)

[<DllImport("libMLIR-C.so", CallingConvention = CallingConvention.Cdecl)>]
extern void mlirRegisterAllDialects(nativeint context)

[<DllImport("libMLIR-C.so", CallingConvention = CallingConvention.Cdecl)>]
extern void mlirContextGetOrLoadDialect(nativeint context, nativeint dialectNamespace)

[<DllImport("libMLIR-C.so", CallingConvention = CallingConvention.Cdecl)>]
extern bool mlirContextIsRegisteredOperation(nativeint context, nativeint operationName)

[<DllImport("libMLIR-C.so", CallingConvention = CallingConvention.Cdecl)>]
extern nativeint mlirStringRefCreateFromCString(string str)

[<DllImport("libMLIR-C.so", CallingConvention = CallingConvention.Cdecl)>]
extern nativeint mlirModuleCreateEmpty(nativeint location)

[<DllImport("libMLIR-C.so", CallingConvention = CallingConvention.Cdecl)>]
extern nativeint mlirLocationUnknownGet(nativeint context)

/// Represents an MLIR context which holds dialect registrations and type information
type MLIRContext = {
    ContextHandle: nativeint
    IsInitialized: bool
    RegisteredDialects: MLIRDialect list
    ModuleCount: int
}

/// Maps dialect enum to string names for MLIR C API
let dialectToString (dialect: MLIRDialect) : string =
    match dialect with
    | Standard -> "std"
    | LLVM -> "llvm"
    | Func -> "func"
    | Arith -> "arith"
    | SCF -> "scf"
    | MLIRDialect.MemRef -> "memref"
    | MLIRDialect.Index -> "index"
    | Affine -> "affine"
    | MLIRDialect.Builtin -> "builtin"

/// Registers a specific dialect with the MLIR context
let private registerSingleDialect (contextHandle: nativeint) (dialect: MLIRDialect) : unit =
    let dialectName = dialectToString dialect
    let dialectNameRef = mlirStringRefCreateFromCString(dialectName)
    mlirContextGetOrLoadDialect(contextHandle, dialectNameRef)

/// Creates a new MLIR context with all required dialects registered
let createContext() : MLIRContext =
    let contextHandle = mlirContextCreate()
    
    if contextHandle = 0n then
        failwith "Failed to create MLIR context"
    
    // Register all dialects that MLIR provides
    mlirRegisterAllDialects(contextHandle)
    
    let requiredDialects = [
        Standard
        LLVM
        Func
        Arith
        MLIRDialect.MemRef
        Affine
        SCF
    ]
    
    // Explicitly load the dialects we need
    requiredDialects |> List.iter (registerSingleDialect contextHandle)
    
    { 
        ContextHandle = contextHandle
        IsInitialized = true
        RegisteredDialects = requiredDialects
        ModuleCount = 0
    }

/// Adds a dialect to an existing context
let registerDialect (context: MLIRContext) (dialect: MLIRDialect) : MLIRContext =
    if context.RegisteredDialects |> List.contains dialect then
        context
    else
        registerSingleDialect context.ContextHandle dialect
        { context with RegisteredDialects = dialect :: context.RegisteredDialects }

/// Checks if a dialect is registered in the context
let isDialectRegistered (context: MLIRContext) (dialect: MLIRDialect) : bool =
    context.RegisteredDialects |> List.contains dialect

/// Verifies an operation is available in the context
let isOperationRegistered (context: MLIRContext) (operationName: string) : bool =
    let opNameRef = mlirStringRefCreateFromCString(operationName)
    mlirContextIsRegisteredOperation(context.ContextHandle, opNameRef)

/// Gets the list of registered dialects
let getRegisteredDialects (context: MLIRContext) : MLIRDialect list =
    context.RegisteredDialects

/// Increments the module count for tracking
let incrementModuleCount (context: MLIRContext) : MLIRContext =
    { context with ModuleCount = context.ModuleCount + 1 }

/// Validates that the context is properly initialized
let validateContext (context: MLIRContext) : bool =
    context.IsInitialized && 
    context.ContextHandle <> 0n &&
    not (List.isEmpty context.RegisteredDialects)

/// Creates an empty MLIR module in this context
let createEmptyModule (context: MLIRContext) : nativeint =
    let unknownLoc = mlirLocationUnknownGet(context.ContextHandle)
    mlirModuleCreateEmpty(unknownLoc)

/// Gets context statistics for debugging
let getContextStats (context: MLIRContext) : string =
    let dialectNames = context.RegisteredDialects |> List.map dialectToString |> String.concat ", "
    sprintf "MLIR Context Stats:\n  Handle: 0x%x\n  Initialized: %b\n  Dialects: [%s]\n  Modules: %d"
        (int64 context.ContextHandle)
        context.IsInitialized
        dialectNames
        context.ModuleCount

/// Disposes of MLIR context and frees resources
let disposeContext (context: MLIRContext) : unit =
    if context.ContextHandle <> 0n then
        mlirContextDestroy(context.ContextHandle)

/// Creates a context with specific dialects only
let createContextWithDialects (dialects: MLIRDialect list) : MLIRContext =
    let contextHandle = mlirContextCreate()
    
    if contextHandle = 0n then
        failwith "Failed to create MLIR context"
    
    mlirRegisterAllDialects(contextHandle)
    dialects |> List.iter (registerSingleDialect contextHandle)
    
    { 
        ContextHandle = contextHandle
        IsInitialized = true
        RegisteredDialects = dialects
        ModuleCount = 0
    }

/// Verifies all required dialects are available for Firefly compilation
let verifyFireflyDialects (context: MLIRContext) : bool =
    let requiredForFirefly = [Standard; LLVM; Func; Arith; MLIRDialect.MemRef]
    let requiredOps = [
        "func.func"
        "func.return"
        "func.call"
        "arith.constant"
        "arith.addi"
        "arith.subi"
        "arith.muli"
        "llvm.func"
        "llvm.return"
        "memref.alloc"
        "memref.load"
        "memref.store"
    ]
    
    let dialectsRegistered = requiredForFirefly |> List.forall (isDialectRegistered context)
    let operationsAvailable = requiredOps |> List.forall (isOperationRegistered context)
    
    dialectsRegistered && operationsAvailable

/// Initializes MLIR subsystem and returns a ready-to-use context
let initializeMLIR() : MLIRContext =
    let context = createContext()
    if not (verifyFireflyDialects context) then
        disposeContext context
        failwith "Failed to initialize required MLIR dialects for Firefly"
    context

/// Gets a dialect-specific operation from the global registry
let getDialectOperation (dialectOp: string) : DialectOperation option =
    let parts = dialectOp.Split('.')
    match parts with
    | [|dialectStr; opName|] ->
        // Map string back to dialect enum
        let dialectOpt = 
            match dialectStr with
            | "std" -> Some Standard
            | "llvm" -> Some LLVM
            | "func" -> Some Func
            | "arith" -> Some Arith
            | "scf" -> Some SCF
            | "memref" -> Some MLIRDialect.MemRef
            | "index" -> Some MLIRDialect.Index
            | "affine" -> Some Affine
            | "builtin" -> Some MLIRDialect.Builtin
            | _ -> None
            
        dialectOpt |> Option.bind (fun dialect ->
            getDialectOperations dialect
            |> List.tryFind (fun op -> op.OpName = dialectOp))
    | _ -> None

/// Ensures a specific operation is available in the context
let ensureOperationAvailable (context: MLIRContext) (dialectOp: string) : MLIRContext =
    match getDialectOperation dialectOp with
    | Some op ->
        if not (isDialectRegistered context op.Dialect) then
            registerDialect context op.Dialect
        else
            context
    | None -> 
        failwithf "Unknown operation: %s" dialectOp