module CLI.Configurations.ProjectConfig

open System
open System.IO
open Core.XParsec.Foundation

/// Dependency binding configuration
type DependencyBinding = {
    Name: string
    Version: string
    Binding: string
    SelectiveLinking: bool
}

/// Compilation settings
type CompilationConfig = {
    RequireStaticMemory: bool
    EliminateClosures: bool
    OptimizationLevel: string
    StackLimit: int option
    KeepIntermediates: bool
    UseLTO: bool
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
        EliminateClosures = true
        OptimizationLevel = "default"
        StackLimit = None
        KeepIntermediates = false
        UseLTO = false
    }
}

/// TOML parsing and configuration handling without relying on excessive XParsec
module TOMLParser =
    /// Simple TOML key-value pair type
    type TOMLValue =
        | String of string
        | Int of int
        | Float of float
        | Bool of bool
        | Array of TOMLValue list
        | Table of Map<string, TOMLValue>
    
    /// Parses a TOML string into a structured format
    let parse (tomlString: string) : Map<string, TOMLValue> =
        // Initialize result map
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
                        // Boolean
                        if valueStr = "true" then TOMLValue.Bool true
                        elif valueStr = "false" then TOMLValue.Bool false
                        // Integer
                        elif Int32.TryParse(valueStr, ref 0) then 
                            TOMLValue.Int(Int32.Parse(valueStr))
                        // Float 
                        elif Double.TryParse(valueStr, ref 0.0) then
                            TOMLValue.Float(Double.Parse(valueStr))
                        // String with quotes
                        elif (valueStr.StartsWith("\"") && valueStr.EndsWith("\"")) ||
                             (valueStr.StartsWith("'") && valueStr.EndsWith("'")) then
                            TOMLValue.String(valueStr.Substring(1, valueStr.Length - 2))
                        // Unquoted string
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
        | Some (TOMLValue.Float value) -> string value
        | Some (TOMLValue.Bool value) -> string value
        | _ -> defaultValue
    
    /// Gets a boolean value from parsed TOML
    let getBool (map: Map<string, TOMLValue>) (path: string) (defaultValue: bool) : bool =
        match Map.tryFind path map with
        | Some (TOMLValue.Bool value) -> value
        | Some (TOMLValue.String "true") -> true
        | Some (TOMLValue.String "false") -> false
        | Some (TOMLValue.Int 0) -> false
        | Some (TOMLValue.Int _) -> true
        | _ -> defaultValue
    
    /// Gets an integer value from parsed TOML
    let getInt (map: Map<string, TOMLValue>) (path: string) : int option =
        match Map.tryFind path map with
        | Some (TOMLValue.Int value) -> Some value
        | Some (TOMLValue.String s) -> 
            match Int32.TryParse(s) with
            | true, value -> Some value
            | _ -> None
        | _ -> None
    
    /// Gets a string value with a specific prefix
    let getStringsWithPrefix (map: Map<string, TOMLValue>) (prefix: string) : (string * string) list =
        map
        |> Map.toList
        |> List.choose (fun (key, value) ->
            if key.StartsWith(prefix) then
                match value with
                | TOMLValue.String str -> Some (key.Substring(prefix.Length), str)
                | _ -> None
            else None)

/// Parses a TOML configuration file using simplified approach
let parseConfigFile (filePath: string) : CompilerResult<FireflyConfig> =
    try
        if not (File.Exists(filePath)) then
            Success defaultConfig
        else
            // Read and parse TOML file
            let tomlString = File.ReadAllText(filePath)
            let tomlMap = TOMLParser.parse tomlString
            
            // Extract package information
            let packageName = TOMLParser.getString tomlMap "package.name" defaultConfig.PackageName
            let version = TOMLParser.getString tomlMap "package.version" defaultConfig.Version
            
            // Extract compilation settings
            let requireStaticMemory = 
                TOMLParser.getBool tomlMap "compilation.require_static_memory" 
                    defaultConfig.Compilation.RequireStaticMemory
                    
            let eliminateClosures = 
                TOMLParser.getBool tomlMap "compilation.eliminate_closures" 
                    defaultConfig.Compilation.EliminateClosures
                    
            let optimizationLevel = 
                TOMLParser.getString tomlMap "compilation.optimize" 
                    defaultConfig.Compilation.OptimizationLevel
                    
            let stackLimit = TOMLParser.getInt tomlMap "compilation.max_stack_size"
            
            let keepIntermediates = 
                TOMLParser.getBool tomlMap "compilation.keep_intermediates" 
                    defaultConfig.Compilation.KeepIntermediates
                    
            let ltoStr = TOMLParser.getString tomlMap "compilation.lto" "false"
            let useLTO = ltoStr.ToLowerInvariant() = "full" || ltoStr.ToLowerInvariant() = "true"
            
            // Extract dependencies
            let dependencyPrefix = "dependencies."
            let dependencyPaths = 
                TOMLParser.getStringsWithPrefix tomlMap dependencyPrefix
                |> List.map fst
                |> List.distinct
                |> List.map (fun depName ->
                    let version = TOMLParser.getString tomlMap (dependencyPrefix + depName + ".version") "0.1.0"
                    let binding = TOMLParser.getString tomlMap (dependencyPrefix + depName + ".binding") "static"
                    let selectiveLinking = 
                        TOMLParser.getBool tomlMap (dependencyPrefix + depName + ".selective_linking") true
                    
                    {
                        Name = depName
                        Version = version
                        Binding = binding
                        SelectiveLinking = selectiveLinking
                    })
            
            // Build final configuration
            let config = {
                PackageName = packageName
                Version = version
                Dependencies = dependencyPaths
                Compilation = {
                    RequireStaticMemory = requireStaticMemory
                    EliminateClosures = eliminateClosures
                    OptimizationLevel = optimizationLevel
                    StackLimit = stackLimit
                    KeepIntermediates = keepIntermediates
                    UseLTO = useLTO
                }
            }
            
            Success config
    with ex ->
        CompilerFailure [SyntaxError(
            { Line = 0; Column = 0; File = filePath; Offset = 0 },
            sprintf "Error reading configuration file: %s" ex.Message,
            ["config parsing"])]

/// Validates the configuration for consistency and correctness
let validateConfig (config: FireflyConfig) : CompilerResult<FireflyConfig> =
    // Validate package name
    if String.IsNullOrWhiteSpace(config.PackageName) then
        CompilerFailure [
            SyntaxError(
                { Line = 0; Column = 0; File = ""; Offset = 0 },
                "Package name cannot be empty",
                ["config validation"]
            )
        ]
    
    // Validate version format (basic check)
    elif not (config.Version.Contains(".")) then
        CompilerFailure [
            SyntaxError(
                { Line = 0; Column = 0; File = ""; Offset = 0 },
                "Version must contain at least one dot (e.g., '1.0.0')",
                ["config validation"]
            )
        ]
    
    // Validate optimization level
    elif not (["none"; "less"; "default"; "aggressive"; "size"; "sizemin"] 
              |> List.contains (config.Compilation.OptimizationLevel.ToLowerInvariant())) then
        let errorMsg = sprintf "Invalid optimization level: %s" config.Compilation.OptimizationLevel
        CompilerFailure [
            SyntaxError(
                { Line = 0; Column = 0; File = ""; Offset = 0 },
                errorMsg,
                ["config validation"]
            )
        ]
    
    // Validate stack limit
    elif config.Compilation.StackLimit.IsSome && config.Compilation.StackLimit.Value <= 0 then
        CompilerFailure [
            SyntaxError(
                { Line = 0; Column = 0; File = ""; Offset = 0 },
                "Stack limit must be positive",
                ["config validation"]
            )
        ]
    
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
eliminate_closures = true
optimize = "default"
keep_intermediates = false
lto = "false"

[dependencies]
# Add your dependencies here
# [dependencies.crypto_lib]
# version = "0.2.0"
# binding = "static"
# selective_linking = true
"""
    
    try
        File.WriteAllText(filePath, defaultToml.Trim())
        Success ()
    with
    | ex ->
        let errorMsg = sprintf "Failed to write default config to %s" filePath
        CompilerFailure [
            InternalError(
                "config generation", 
                errorMsg,
                Some ex.Message
            )
        ]