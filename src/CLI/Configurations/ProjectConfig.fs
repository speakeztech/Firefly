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

/// Parses a TOML configuration file
let parseConfigFile (filePath: string) : FireflyConfig =
    if not (File.Exists(filePath)) then
        defaultConfig
    else
        try
            let tomlString = File.ReadAllText(filePath)
            let model = Toml.ToModel(tomlString)

            // Parse package section
            let packageTable = model.GetTable("package")
            let packageName = 
                if packageTable <> null && packageTable.ContainsKey("name") then
                    packageTable.["name"].ToString()
                else
                    defaultConfig.PackageName

            let version =
                if packageTable <> null && packageTable.ContainsKey("version") then
                    packageTable.["version"].ToString()
                else
                    defaultConfig.Version

            // Parse compilation section
            let compilationTable = model.GetTable("compilation")
            let compilation = 
                if compilationTable = null then
                    defaultConfig.Compilation
                else
                    {
                        RequireStaticMemory = 
                            if compilationTable.ContainsKey("require_static_memory") then
                                compilationTable.["require_static_memory"] :?> bool
                            else 
                                defaultConfig.Compilation.RequireStaticMemory
                        EliminateClosures = 
                            if compilationTable.ContainsKey("eliminate_closures") then
                                compilationTable.["eliminate_closures"] :?> bool
                            else 
                                defaultConfig.Compilation.EliminateClosures
                        OptimizationLevel = 
                            if compilationTable.ContainsKey("optimize") then
                                compilationTable.["optimize"].ToString()
                            else 
                                defaultConfig.Compilation.OptimizationLevel
                        StackLimit = 
                            if compilationTable.ContainsKey("max_stack_size") then
                                Some(Convert.ToInt32(compilationTable.["max_stack_size"]))
                            else 
                                defaultConfig.Compilation.StackLimit
                        KeepIntermediates = 
                            if compilationTable.ContainsKey("keep_intermediates") then
                                compilationTable.["keep_intermediates"] :?> bool
                            else 
                                defaultConfig.Compilation.KeepIntermediates
                        UseLTO = 
                            if compilationTable.ContainsKey("lto") then
                                compilationTable.["lto"].ToString().ToLowerInvariant() = "full"
                            else 
                                defaultConfig.Compilation.UseLTO
                    }

            // Parse dependencies
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