/// TimeTest - Test timestamp
module TimeTest

open Alloy
open Alloy.Console
open Alloy.Time

[<EntryPoint>]
let main argv =
    let now = currentUnixTimestamp ()
    Write "Unix timestamp: "
    Write (Alloy.Text.Format.int64ToString now)
    WriteLine ""
    0
