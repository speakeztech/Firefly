module CLI.Configurations.ProjectConfig

open System
open System.IO
open Tomlyn
open Tomlyn.Model

/// Represents binding configuration for a dependency
type DependencyBinding = {
    Name: string
    Version: string
    Binding: string
    SelectiveLinking: bool
}

/// Represents compilation settings
type CompilationConfig = {
    RequireStaticMemory: bool
    EliminateClosures: bool
    OptimizationLevel: string
    StackLimit: int option
    KeepIntermediates: bool
    UseLTO: bool
}

/// Represents a complete Firefly project configuration
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

/// Helper function to safely get a string value from a TOML table
let private getStringValue (table: TomlTable) (key: string) (defaultValue: string) : string =
    if table.ContainsKey(key) then
        match table.[key] with
        | :? string as str -> str
        | obj -> obj.ToString()
    else
        defaultValue

/// Helper function to safely get a boolean value from a TOML table
let private getBoolValue (table: TomlTable) (key: string) (defaultValue: bool) : bool =
    if table.ContainsKey(key) then
        match table.[key] with
        | :? bool as b -> b
        | :? string as str -> String.Equals(str, "true", StringComparison.OrdinalIgnoreCase)
        | _ -> defaultValue
    else
        defaultValue

/// Helper function to safely get an integer value from a TOML table
let private getIntValue (table: TomlTable) (key: string) (defaultValue: int option) : int option =
    if table.ContainsKey(key) then
        match table.[key] with
        | :? int as i -> Some i
        | :? int64 as i64 -> Some (int i64)
        | :? string as str -> 
            match Int32.TryParse(str) with
            | (true, value) -> Some value
            | _ -> defaultValue
        | _ -> defaultValue
    else
        defaultValue

/// Helper function to safely get a nested table from a TOML table
let private getTable (rootTable: TomlTable) (key: string) : TomlTable option =
    if rootTable.ContainsKey(key) then
        match rootTable.[key] with
        | :? TomlTable as table -> Some table
        | _ -> None
    else
        None

/// Parses a TOML configuration file
let parseConfigFile (filePath: string) : FireflyConfig =
    if not (File.Exists(filePath)) then
        defaultConfig
    else
        try
            let tomlString = File.ReadAllText(filePath)
            let model = Toml.ToModel(tomlString)

            // Parse package section
            let packageTable = getTable model "package"
            let packageName = 
                match packageTable with
                | Some table -> getStringValue table "name" defaultConfig.PackageName
                | None -> defaultConfig.PackageName

            let version =
                match packageTable with
                | Some table -> getStringValue table "version" defaultConfig.Version
                | None -> defaultConfig.Version

            // Parse compilation section
            let compilationTable = getTable model "compilation"
            let compilation = 
                match compilationTable with
                | None -> defaultConfig.Compilation
                | Some table ->
                    {
                        RequireStaticMemory = getBoolValue table "require_static_memory" defaultConfig.Compilation.RequireStaticMemory
                        EliminateClosures = getBoolValue table "eliminate_closures" defaultConfig.Compilation.EliminateClosures
                        OptimizationLevel = getStringValue table "optimize" defaultConfig.Compilation.OptimizationLevel
                        StackLimit = getIntValue table "max_stack_size" defaultConfig.Compilation.StackLimit
                        KeepIntermediates = getBoolValue table "keep_intermediates" defaultConfig.Compilation.KeepIntermediates
                        UseLTO = 
                            let ltoValue = getStringValue table "lto" "false"
                            ltoValue.ToLowerInvariant() = "full"
                    }

            // Parse dependencies (placeholder for now)
            let dependencies = []

            { 
                PackageName = packageName
                Version = version
                Dependencies = dependencies
                Compilation = compilation
            }
        with
        | ex ->
            printfn "Error parsing configuration: %s" ex.Message
            defaultConfig