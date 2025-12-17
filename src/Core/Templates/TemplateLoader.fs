module Core.Templates.TemplateLoader

open System
open System.IO
open Core.Templates.TemplateTypes

/// TOML parsing result
type TomlResult<'T> =
    | Success of 'T
    | ParseError of message: string

/// Simplified TOML parser for templates
/// Note: In production, use a proper TOML library like Tomlyn
module TomlParser =
    open System.Collections.Generic
    open System.Text.RegularExpressions
    
    type TomlValue =
        | String of string
        | Integer of int64
        | Float of float
        | Boolean of bool
        | Array of TomlValue list
        | Table of Map<string, TomlValue>
    
    let rec private parseValue (value: string) : TomlValue =
        let trimmed = value.Trim()
        
        // String
        if trimmed.StartsWith("\"") && trimmed.EndsWith("\"") then
            String (trimmed.Substring(1, trimmed.Length - 2))
        // Boolean
        elif trimmed = "true" then Boolean true
        elif trimmed = "false" then Boolean false
        // Array
        elif trimmed.StartsWith("[") && trimmed.EndsWith("]") then
            let inner = trimmed.Substring(1, trimmed.Length - 2)
            let items = 
                inner.Split(',')
                |> Array.map (fun s -> parseValue (s.Trim()))
                |> List.ofArray
            Array items
        // Integer or Float
        else
            match System.Int64.TryParse(trimmed) with
            | true, i -> Integer i
            | false, _ ->
                match System.Double.TryParse(trimmed) with
                | true, f -> Float f
                | false, _ -> String trimmed
    
    let parse (content: string) : TomlResult<Map<string, TomlValue>> =
        try
            let lines = content.Split('\n')
            let mutable currentTable = ""
            let mutable result = Map.empty<string, TomlValue>
            let mutable tables = Map.empty<string, Map<string, TomlValue>>
            
            for line in lines do
                let trimmed = line.Trim()
                
                // Skip comments and empty lines
                if trimmed.StartsWith("#") || String.IsNullOrWhiteSpace(trimmed) then
                    ()
                // Table header
                elif trimmed.StartsWith("[") && trimmed.EndsWith("]") then
                    currentTable <- trimmed.Substring(1, trimmed.Length - 2)
                    if not (tables.ContainsKey(currentTable)) then
                        tables <- tables.Add(currentTable, Map.empty)
                // Key-value pair
                elif trimmed.Contains("=") then
                    let parts = trimmed.Split('=', 2)
                    let key = parts.[0].Trim()
                    let value = parseValue parts.[1]
                    
                    if String.IsNullOrEmpty(currentTable) then
                        result <- result.Add(key, value)
                    else
                        let tableContent = Map.find currentTable tables
                        tables <- tables.Add(currentTable, tableContent.Add(key, value))
            
            // Combine tables into result
            for KeyValue(tableName, tableContent) in tables do
                result <- result.Add(tableName, Table tableContent)
            
            Success result
        with ex ->
            ParseError $"Failed to parse TOML: {ex.Message}"

/// Convert TOML to platform template
let private tomlToTemplate (toml: Map<string, TomlParser.TomlValue>) : TomlResult<PlatformTemplate> =
    try
        // Extract platform info
        let platform = 
            match Map.tryFind "platform" toml with
            | Some (TomlParser.Table p) ->
                {
                    Family = 
                        match Map.tryFind "family" p with
                        | Some (TomlParser.String s) -> s
                        | _ -> ""
                    Architecture = 
                        match Map.tryFind "architecture" p with
                        | Some (TomlParser.String s) -> s
                        | _ -> ""
                    Variant = 
                        match Map.tryFind "variant" p with
                        | Some (TomlParser.String s) -> Some s
                        | _ -> None
                }
            | _ -> { Family = ""; Architecture = ""; Variant = None }
        
        // Extract memory regions
        let regions = 
            toml
            |> Map.toList
            |> List.choose (fun (key, value) ->
                if key.StartsWith("region.") then
                    match value with
                    | TomlParser.Table r ->
                        let name = key.Substring(7)  // Remove "region."
                        Some {
                            Name = name
                            BaseAddress = 
                                match Map.tryFind "base" r with
                                | Some (TomlParser.String s) when s.StartsWith("0x") ->
                                    Some (System.Convert.ToUInt64(s.Substring(2), 16))
                                | Some (TomlParser.Integer i) -> Some (uint64 i)
                                | _ -> None
                            Size = 
                                match Map.tryFind "size" r with
                                | Some (TomlParser.String s) when s.EndsWith("KB") ->
                                    System.Int32.Parse(s.Replace("KB", "")) * 1024
                                | Some (TomlParser.String s) when s.EndsWith("MB") ->
                                    System.Int32.Parse(s.Replace("MB", "")) * 1024 * 1024
                                | Some (TomlParser.Integer i) -> int i
                                | _ -> 0
                            Attributes = 
                                match Map.tryFind "attributes" r with
                                | Some (TomlParser.Array attrs) ->
                                    attrs |> List.choose (function
                                        | TomlParser.String s -> Some s
                                        | _ -> None
                                    )
                                | _ -> []
                            Access = ReadWrite  // Default
                        }
                    | _ -> None
                else None
            )
        
        // Extract capabilities
        let capabilities = 
            match Map.tryFind "capabilities" toml with
            | Some (TomlParser.Array caps) ->
                caps |> List.choose (function
                    | TomlParser.String "trustzone" -> Some TrustZone
                    | TomlParser.String "crypto" -> Some CryptoAcceleration
                    | TomlParser.String s when s.StartsWith("cxl") -> 
                        Some (CXL (s.Replace("cxl", "").Trim()))
                    | _ -> None
                )
            | _ -> []
        
        // Create template
        Success {
            Name = 
                match Map.tryFind "name" toml with
                | Some (TomlParser.String s) -> s
                | _ -> "unnamed"
            Version = 
                match Map.tryFind "version" toml with
                | Some (TomlParser.String s) -> s
                | _ -> "1.0"
            Platform = platform
            MemoryRegions = regions
            Capabilities = capabilities
            Profiles = []  // Profile parsing not yet implemented - templates work without profiles
        }
    with ex ->
        ParseError $"Failed to convert TOML to template: {ex.Message}"

/// Load template from TOML file
let loadFromFile (path: string) : TomlResult<PlatformTemplate> =
    try
        if not (File.Exists(path)) then
            ParseError $"Template file not found: {path}"
        else
            let content = File.ReadAllText(path)
            match TomlParser.parse content with
            | Success toml -> tomlToTemplate toml
            | ParseError msg -> ParseError msg
    with ex ->
        ParseError $"Failed to load template: {ex.Message}"

/// Load template from string
let loadFromString (content: string) : TomlResult<PlatformTemplate> =
    match TomlParser.parse content with
    | Success toml -> tomlToTemplate toml
    | ParseError msg -> ParseError msg

/// Example TOML template content
let exampleToml = """
name = "stm32l5"
version = "1.0"

[platform]
family = "STM32"
architecture = "ARMv8-M"
variant = "STM32L5"

[region.tcm]
base = "0x20000000"
size = "64KB"
attributes = ["fast", "secure"]

[region.sram1]
base = "0x20010000"
size = "192KB"
attributes = ["cached"]

[region.flash]
base = "0x08000000"
size = "512KB"
attributes = ["persistent", "secure"]

capabilities = ["trustzone", "crypto", "dma"]

[profile.secure_iot]
requires = ["trustzone", "crypto"]
allocation_strategy = "static_pools"
prefer_regions = ["tcm"]
"""

/// Load all templates from directory
let loadTemplatesFromDirectory (dir: string) : (string * PlatformTemplate) list =
    if not (Directory.Exists(dir)) then []
    else
        Directory.GetFiles(dir, "*.toml")
        |> Array.choose (fun file ->
            match loadFromFile file with
            | Success template -> 
                let name = Path.GetFileNameWithoutExtension(file)
                Some (name, template)
            | ParseError msg ->
                eprintfn $"Failed to load {file}: {msg}"
                None
        )
        |> List.ofArray

/// Save template to TOML file
let saveToFile (path: string) (template: PlatformTemplate) =
    let toml = System.Text.StringBuilder()
    
    // Basic info
    toml.AppendLine($"name = \"{template.Name}\"") |> ignore
    toml.AppendLine($"version = \"{template.Version}\"") |> ignore
    toml.AppendLine() |> ignore
    
    // Platform
    toml.AppendLine("[platform]") |> ignore
    toml.AppendLine($"family = \"{template.Platform.Family}\"") |> ignore
    toml.AppendLine($"architecture = \"{template.Platform.Architecture}\"") |> ignore
    match template.Platform.Variant with
    | Some v -> toml.AppendLine($"variant = \"{v}\"") |> ignore
    | None -> ()
    toml.AppendLine() |> ignore
    
    // Memory regions
    for region in template.MemoryRegions do
        toml.AppendLine($"[region.{region.Name.ToLower()}]") |> ignore
        match region.BaseAddress with
        | Some addr -> toml.AppendLine($"base = \"0x{addr:X}\"") |> ignore
        | None -> ()
        
        let sizeStr = 
            if region.Size >= 1024 * 1024 then
                $"{region.Size / (1024 * 1024)}MB"
            elif region.Size >= 1024 then
                $"{region.Size / 1024}KB"
            else
                $"{region.Size}"
        toml.AppendLine($"size = \"{sizeStr}\"") |> ignore
        
        if not region.Attributes.IsEmpty then
            let attrs = region.Attributes |> List.map (fun a -> $"\"{a}\"") |> String.concat ", "
            toml.AppendLine($"attributes = [{attrs}]") |> ignore
        toml.AppendLine() |> ignore
    
    // Capabilities
    if not template.Capabilities.IsEmpty then
        let caps = 
            template.Capabilities 
            |> List.map (function
                | TrustZone -> "\"trustzone\""
                | CryptoAcceleration -> "\"crypto\""
                | CXL v -> $"\"cxl{v}\""
                | NUMA n -> $"\"numa{n}\""
                | _ -> "\"\""
            )
            |> String.concat ", "
        toml.AppendLine($"capabilities = [{caps}]") |> ignore
    
    File.WriteAllText(path, toml.ToString())

/// Initialize template registry with built-in and custom templates
let initializeRegistry (customDir: string option) =
    let builtIn = 
        [
            "stm32l5", CommonTemplates.stm32l5
            "apple-m2", CommonTemplates.appleM2
            "x86-server", CommonTemplates.x86Server
        ]
        |> Map.ofList
    
    let custom = 
        match customDir with
        | Some dir -> 
            loadTemplatesFromDirectory dir |> Map.ofList
        | None -> 
            Map.empty
    
    {
        Templates = builtIn
        CustomTemplates = custom
    }