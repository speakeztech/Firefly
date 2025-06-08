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

type OakDeclaration =
    | FunctionDecl of name: string * params': (string * OakType) list * returnType: OakType * body: OakExpression
    | TypeDecl of name: string * oakType: OakType
    | EntryPoint of OakExpression

type OakModule = {
    Name: string
    Declarations: OakDeclaration list
}

type OakProgram = {
    Modules: OakModule list
}
