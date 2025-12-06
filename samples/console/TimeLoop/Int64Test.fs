/// Int64Test - Test int64 printing
module Int64Test

open Alloy
open Alloy.Console

[<EntryPoint>]
let main argv =
    let x = 10000000L
    Write "Value: "
    Write (Alloy.Text.Format.int64ToString x)
    WriteLine ""
    0
