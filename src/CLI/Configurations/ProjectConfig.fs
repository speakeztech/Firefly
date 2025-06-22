module CLI.Configurations.ProjectConfig

open System
open System.IO
open Core.XParsec.Foundation

/// Dependency binding configuration
type DependencyBinding = {
    Name: string
    Version: string
    Binding: string
}

/// Compilation settings
type CompilationConfig = {
    RequireStaticMemory: bool
    OptimizationLevel: string
    StackLimit: int option
    KeepIntermediates: bool
}

/// Complete Firefly project configuration
type FireflyConfig = {
    PackageName: string
    Version: string
    Dependencies: DependencyBinding list
    Compilation: CompilationConfig
}

/// Default configuration values
let defaultConfig = {
    PackageName = "unknown"
    Version = "0.1.0"
    Dependencies = []
    Compilation = {
        RequireStaticMemory = true
        OptimizationLevel = "default"
        StackLimit = None
        KeepIntermediates = false
    }
}

/// Simple key-value pair type for TOML parsing
type TOMLValue =
    | String of string
    | Int of int
    | Float of float
    | Bool of bool
    | Table of Map<string, TOMLValue>

/// Parse a TOML string into a structured format
let parseTOML (tomlString: string) : Map<string, TOMLValue> =
    let mutable result = Map.empty
    let mutable currentSection = []
    
    // Process each line
    let lines = 
        tomlString.Split('\n')
        |> Array.map (fun line -> line.Trim())
        |> Array.filter (fun line -> 
            not (String.IsNullOrWhiteSpace(line)) && not (line.StartsWith("#")))
    
    for line in lines do
        // Section header: [section] or [section.subsection]
        if line.StartsWith("[") && line.EndsWith("]") then
            let sectionName = line.Substring(1, line.Length - 2).Trim()
            currentSection <- sectionName.Split('.') |> Array.toList
        
        // Key-value pair: key = value
        elif line.Contains("=") then
            let parts = line.Split('=', 2)
            if parts.Length = 2 then
                let key = parts.[0].Trim()
                let valueStr = parts.[1].Trim()
                
                // Parse value based on type
                let value =
                    if valueStr = "true" then TOMLValue.Bool true
                    elif valueStr = "false" then TOMLValue.Bool false
                    elif Int32.TryParse(valueStr, ref 0) then TOMLValue.Int(Int32.Parse(valueStr))
                    elif (valueStr.StartsWith("\"") && valueStr.EndsWith("\"")) then
                        TOMLValue.String(valueStr.Substring(1, valueStr.Length - 2))
                    else TOMLValue.String(valueStr)
                
                // Add to result map with proper section path
                let path = String.concat "." (currentSection @ [key])
                result <- Map.add path value result
    
    result

/// Gets a string value from parsed TOML
let getString (map: Map<string, TOMLValue>) (path: string) (defaultValue: string) : string =
    match Map.tryFind path map with
    | Some (TOMLValue.String value) -> value
    | Some (TOMLValue.Int value) -> string value
    | Some (TOMLValue.Bool value) -> string value
    | _ -> defaultValue

/// Gets a boolean value from parsed TOML
let getBool (map: Map<string, TOMLValue>) (path: string) (defaultValue: bool) : bool =
    match Map.tryFind path map with
    | Some (TOMLValue.Bool value) -> value
    | Some (TOMLValue.String "true") -> true
    | Some (TOMLValue.String "false") -> false
    | _ -> defaultValue

/// Gets an integer value from parsed TOML
let getInt (map: Map<string, TOMLValue>) (path: string) : int option =
    match Map.tryFind path map with
    | Some (TOMLValue.Int value) -> Some value
    | _ -> None

/// Parses a TOML configuration file
let parseConfigFile (filePath: string) : CompilerResult<FireflyConfig> =
    try
        if not (File.Exists(filePath)) then
            Success defaultConfig
        else
            // Read and parse TOML file
            let tomlString = File.ReadAllText(filePath)
            let tomlMap = parseTOML tomlString
            
            // Extract package information
            let packageName = getString tomlMap "package.name" defaultConfig.PackageName
            let version = getString tomlMap "package.version" defaultConfig.Version
            
            // Extract compilation settings
            let requireStaticMemory = getBool tomlMap "compilation.require_static_memory" defaultConfig.Compilation.RequireStaticMemory
            let optimizationLevel = getString tomlMap "compilation.optimize" defaultConfig.Compilation.OptimizationLevel
            let stackLimit = getInt tomlMap "compilation.max_stack_size"
            let keepIntermediates = getBool tomlMap "compilation.keep_intermediates" defaultConfig.Compilation.KeepIntermediates
            
            // Build final configuration
            let config = {
                PackageName = packageName
                Version = version
                Dependencies = [] // Simplified - could be expanded as needed
                Compilation = {
                    RequireStaticMemory = requireStaticMemory
                    OptimizationLevel = optimizationLevel
                    StackLimit = stackLimit
                    KeepIntermediates = keepIntermediates
                }
            }
            
            Success config
    with ex ->
        CompilerFailure [SyntaxError(
            { Line = 0; Column = 0; File = filePath; Offset = 0 },
            sprintf "Error reading configuration file: %s" ex.Message,
            ["config parsing"])]

/// Validates configuration for consistency
let validateConfig (config: FireflyConfig) : CompilerResult<FireflyConfig> =
    if String.IsNullOrWhiteSpace(config.PackageName) then
        CompilerFailure [SyntaxError(
            { Line = 0; Column = 0; File = ""; Offset = 0 },
            "Package name cannot be empty",
            ["config validation"])]
    elif config.Compilation.StackLimit.IsSome && config.Compilation.StackLimit.Value <= 0 then
        CompilerFailure [SyntaxError(
            { Line = 0; Column = 0; File = ""; Offset = 0 },
            "Stack limit must be positive",
            ["config validation"])]
    else
        Success config

/// Main entry point for loading and validating configuration
let loadAndValidateConfig (filePath: string) : CompilerResult<FireflyConfig> =
    match parseConfigFile filePath with
    | Success config -> validateConfig config
    | CompilerFailure errors -> CompilerFailure errors

/// Generates a default configuration file
let generateDefaultConfigFile (filePath: string) : CompilerResult<unit> =
    let defaultToml = """
[package]
name = "my_firefly_project"
version = "0.1.0"

[compilation]
require_static_memory = true
optimize = "default"
keep_intermediates = false
"""
    
    try
        File.WriteAllText(filePath, defaultToml.Trim())
        Success ()
    with ex ->
        CompilerFailure [InternalError(
            "config generation", 
            sprintf "Failed to write default config to %s" filePath,
            Some ex.Message)]