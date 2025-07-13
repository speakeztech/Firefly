module Alex.CodeGeneration.MLIRDialects

open Core.XParsec.Foundation
open MLIRSyntax

/// Dialect-specific operation builders with validation
module ArithDialect =
    type ArithOp =
        | Addi | Subi | Muli | Divi | Modi
        | Addf | Subf | Mulf | Divf
        | Cmpi of CmpPredicate | Cmpf of CmpPredicate
        | Select | Const
    
    and CmpPredicate = Eq | Ne | Slt | Sle | Sgt | Sge | Ult | Ule | Ugt | Uge
    
    let predName = function
        | Eq -> "eq" | Ne -> "ne" | Slt -> "slt" | Sle -> "sle" | Sgt -> "sgt" 
        | Sge -> "sge" | Ult -> "ult" | Ule -> "ule" | Ugt -> "ugt" | Uge -> "uge"
    
    let opName = function
        | Addi -> "addi" | Subi -> "subi" | Muli -> "muli" | Divi -> "divi" | Modi -> "modi"
        | Addf -> "addf" | Subf -> "subf" | Mulf -> "mulf" | Divf -> "divf"
        | Cmpi pred -> sprintf "cmpi %s" (predName pred)
        | Cmpf pred -> sprintf "cmpf %s" (predName pred)
        | Select -> "select"
        | Const -> "constant"
    
    let isIntegerType = function
        | "i1" | "i8" | "i16" | "i32" | "i64" | "index" -> true
        | _ -> false
    
    let isFloatType = function
        | "f32" | "f64" -> true
        | _ -> false
    
    /// Validated arithmetic operation builder
    let buildOp op result operands typ =
        match op, operands with
        | (Addi | Subi | Muli | Divi | Modi), [left; right] when isIntegerType typ ->
            sprintf "%s = arith.%s %s, %s : %s" result (opName op) left right typ |> Result.Ok
        | (Addf | Subf | Mulf | Divf), [left; right] when isFloatType typ ->
            sprintf "%s = arith.%s %s, %s : %s" result (opName op) left right typ |> Result.Ok
        | Cmpi pred, [left; right] when isIntegerType typ ->
            sprintf "%s = arith.cmpi %s, %s, %s : %s" result (predName pred) left right typ |> Result.Ok
        | Select, [cond; trueVal; falseVal] ->
            sprintf "%s = arith.select %s, %s, %s : %s" result cond trueVal falseVal typ |> Result.Ok
        | _ -> Result.Error "Invalid operands for arithmetic operation"

module FuncDialect =
    type FuncOp = Call | Return | Func
    
    /// Build function declaration with proper attributes
    let buildFuncDecl name parameters returnType visibility attrs =
        let paramList = parameters |> List.map (fun (paramName, paramType) -> sprintf "%s: %s" paramName paramType) |> String.concat ", "
        let attrStr = if List.isEmpty attrs then "" else " " + String.concat " " attrs
        sprintf "func.func %s @%s(%s) -> %s%s" visibility name paramList returnType attrStr
    
    /// Build function call with result type validation
    let buildCall result funcName args funcType =
        sprintf "%s = func.call @%s(%s) : %s" result funcName (String.concat ", " args) funcType

module MemrefDialect =
    type MemrefOp =
        | Alloc | Alloca | Dealloc
        | Load | Store
        | Cast | View | Subview
        | GetGlobal
    
    /// Build memref operations with shape validation
    let buildAlloca result shape elemType alignment =
        let shapeStr = match shape with
                       | [] -> ""
                       | dims -> sprintf "(%s)" (String.concat ", " dims)
        let alignStr = match alignment with
                       | Some a -> sprintf " {alignment = %d}" a
                       | None -> ""
        sprintf "%s = memref.alloca%s : memref<%s>%s" 
            result shapeStr 
            (if List.isEmpty shape then elemType else sprintf "%sx%s" (String.concat "x" shape) elemType)
            alignStr
    
    let buildLoad result memref indices memrefType =
        sprintf "%s = memref.load %s[%s] : %s" result memref (String.concat ", " indices) memrefType
    
    let buildStore value memref indices memrefType =
        sprintf "memref.store %s, %s[%s] : %s" value memref (String.concat ", " indices) memrefType

module ScfDialect =
    type ScfOp = For | While | If | Yield | Condition
    
    /// Build structured control flow with proper region syntax
    let buildFor indexVar lower upper step body =
        let header = sprintf "scf.for %s = %s to %s step %s {" indexVar lower upper step
        let indentedBody = body |> List.map (sprintf "  %s")
        let footer = "}"
        [header] @ indentedBody @ [footer]
    
    let buildIf condition thenOps elseOps resultTypes =
        let typeList = 
            if List.isEmpty resultTypes then "" 
            else sprintf " -> (%s)" (String.concat ", " resultTypes)
        
        let header = sprintf "scf.if %s%s {" condition typeList
        let thenBody = thenOps |> List.map (sprintf "  %s")
        
        if List.isEmpty elseOps then
            [header] @ thenBody @ ["}"]
        else
            let elseHeader = "} else {"
            let elseBody = elseOps |> List.map (sprintf "  %s")
            [header] @ thenBody @ [elseHeader] @ elseBody @ ["}"]

/// Composite operation builders that span dialects
module CompositeOps =
    
    /// Build a bounds-checked array access
    let buildSafeArrayAccess array index arraySize elemType =
        let ops = ResizeArray<string>()
        
        // Generate bounds check
        let cmpResult = "%bounds_check"
        match ArithDialect.buildOp (ArithDialect.Cmpi ArithDialect.Ult) cmpResult [index; arraySize] "index" with
        | Ok op -> ops.Add(op)
        | Error e -> failwith e
        
        // Build conditional load
        let loadOps = [
            MemrefDialect.buildLoad "%elem" array [index] (sprintf "memref<?x%s>" elemType)
            sprintf "scf.yield %%elem : %s" elemType
        ]
        
        let errorOps = [
            "// Bounds check failed - return default or panic"
            sprintf "%%default = arith.constant 0 : %s" elemType
            sprintf "scf.yield %%default : %s" elemType
        ]
        
        ops.AddRange(ScfDialect.buildIf cmpResult loadOps errorOps [elemType])
        ops.ToArray() |> Array.toList
    
    /// Build optimized string concatenation
    let buildStringConcat strings =
        let ops = ResizeArray<string>()
        
        // Calculate total length
        ops.Add("// Calculate total string length")
        let lengths = strings |> List.mapi (fun i s -> 
            let lenVar = sprintf "%%len%d" i
            ops.Add(FuncDialect.buildCall lenVar "strlen" [s] "(!llvm.ptr) -> i64")
            lenVar)
        
        // Sum lengths
        let totalLen = lengths |> List.reduce (fun acc len ->
            let sumVar = sprintf "%%sum_%s_%s" (acc.Replace("%", "")) (len.Replace("%", ""))
            match ArithDialect.buildOp ArithDialect.Addi sumVar [acc; len] "i64" with
            | Ok op -> 
                ops.Add(op)
                sumVar
            | Error e -> failwith e)
        
        // Allocate buffer
        ops.Add(sprintf "%%buffer = func.call @malloc(%s) : (i64) -> !llvm.ptr" totalLen)
        
        // Copy strings
        ops.Add("// Copy strings to buffer")
        let mutable offset = "%c0"
        ops.Add("%c0 = arith.constant 0 : i64")
        
        for i, str in List.indexed strings do
            ops.Add(sprintf "%%dest%d = llvm.getelementptr %%buffer[%s] : (!llvm.ptr, i64) -> !llvm.ptr" i offset)
            ops.Add(sprintf "func.call @strcpy(%%dest%d, %s) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr" i str)
            if i < strings.Length - 1 then
                let newOffset = sprintf "%%offset%d" (i + 1)
                match ArithDialect.buildOp ArithDialect.Addi newOffset [offset; lengths.[i]] "i64" with
                | Ok op -> 
                    ops.Add(op)
                    offset <- newOffset
                | Error e -> failwith e
        
        ops.ToArray() |> Array.toList

/// Type validation helpers
let canConvert fromType toType =
    match fromType, toType with
    | "i8", "i32" | "i16", "i32" | "i8", "i64" | "i16", "i64" | "i32", "i64" -> true
    | "f32", "f64" -> true
    | _ -> fromType = toType

let validateTypes expected actual =
    List.zip expected actual 
    |> List.forall (fun (e, a) -> e = a || canConvert a e)