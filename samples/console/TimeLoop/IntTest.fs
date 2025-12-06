/// IntTest - Test integer printing
module IntTest

open Alloy
open Alloy.Console

[<EntryPoint>]
let main argv =
    let x = 12345
    Write "Value: "
    Write (Alloy.Text.Format.intToString x)
    WriteLine ""
    0
