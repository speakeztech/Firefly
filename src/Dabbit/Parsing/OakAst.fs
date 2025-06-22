module Dabbit.Parsing.OakAst

/// Represents the Oak AST which is a simplified, canonical representation of F# AST
/// with guarantees about allocation patterns and explicit control flow
type OakType =
    | IntType
    | FloatType
    | BoolType
    | StringType
    | UnitType
    | ArrayType of elementType: OakType
    | FunctionType of paramTypes: OakType list * returnType: OakType
    | StructType of fields: (string * OakType) list
    | UnionType of cases: (string * OakType option) list

/// Represents I/O operation types for external function calls
type IOOperationType =
    | Printf of formatString: string
    | Printfn of formatString: string
    | ReadLine
    | Scanf of formatString: string
    | WriteFile of path: string
    | ReadFile of path: string

/// Represents literal values in the AST
type OakLiteral =
    | IntLiteral of int
    | FloatLiteral of float
    | BoolLiteral of bool
    | StringLiteral of string
    | UnitLiteral
    | ArrayLiteral of OakExpression list

and OakExpression =
    | Literal of OakLiteral
    | Variable of string
    | Application of func: OakExpression * args: OakExpression list
    | Lambda of params': (string * OakType) list * body: OakExpression
    | Let of name: string * value: OakExpression * body: OakExpression
    | IfThenElse of cond: OakExpression * thenExpr: OakExpression * elseExpr: OakExpression
    | Sequential of first: OakExpression * second: OakExpression
    | FieldAccess of target: OakExpression * fieldName: string
    | MethodCall of target: OakExpression * methodName: string * args: OakExpression list
    | IOOperation of ioType: IOOperationType * args: OakExpression list
    | Match of expr: OakExpression * cases: (OakPattern * OakExpression) list  // NEW

and OakPattern =
    | PatternVariable of name: string
    | PatternLiteral of literal: OakLiteral
    | PatternWildcard
    | PatternConstructor of name: string * patterns: OakPattern list  // NEW

type OakDeclaration =
    | FunctionDecl of name: string * params': (string * OakType) list * returnType: OakType * body: OakExpression
    | TypeDecl of name: string * oakType: OakType
    | EntryPoint of OakExpression
    | ExternalDecl of name: string * params': OakType list * returnType: OakType * libraryName: string

type OakModule = {
    Name: string
    Declarations: OakDeclaration list
}

type OakProgram = {
    Modules: OakModule list
}