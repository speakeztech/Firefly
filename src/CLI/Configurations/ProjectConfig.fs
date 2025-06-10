module CLI.Configurations.ProjectConfig

open System
open System.IO
open XParsec
open Core.XParsec.Foundation
open Core.XParsec.Foundation.Combinators
open Core.XParsec.Foundation.CharParsers
open Core.XParsec.Foundation.StringParsers
open Core.XParsec.Foundation.ErrorHandling

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

/// TOML parsing state
type TOMLParsingState = {
    CurrentSection: string list
    ConfigData: Map<string list, Map<string, string>>
    ErrorContext: string list
}

/// TOML value types
type TOMLValue =
    | TOMLString of string
    | TOMLInteger of int
    | TOMLFloat of float
    | TOMLBoolean of bool
    | TOMLArray of TOMLValue list

/// TOML parsing using XParsec combinators - NO EXTERNAL DEPENDENCIES
module TOMLParsers =
    
    /// Parses TOML comments
    let tomlComment : Parser<unit, TOMLParsingState> =
        pchar '#' >>= fun _ ->
        many (satisfy (fun c -> c <> '\n')) >>= fun _ ->
        opt (pchar '\n') >>= fun _ ->
        succeed ()
        |> withErrorContext "TOML comment"
    
    /// Skips whitespace and comments
    let tomlWs : Parser<unit, TOMLParsingState> =
        many (whitespace <|> tomlComment) |>> ignore
    
    /// Parses TOML string literals
    let tomlString : Parser<TOMLValue, TOMLParsingState> =
        let quotedString = 
            between (pchar '"') (pchar '"') (many (satisfy (fun c -> c <> '"'))) |>>
            fun chars -> TOMLString(String(Array.ofList chars))
        
        let singleQuotedString =
            between (pchar '\'') (pchar '\'') (many (satisfy (fun c -> c <> '\''))) |>>
            fun chars -> TOMLString(String(Array.ofList chars))
        
        quotedString <|> singleQuotedString
        |> withErrorContext "TOML string"
    
    /// Parses TOML integer literals
    let tomlInteger : Parser<TOMLValue, TOMLParsingState> =
        let digits = many1 digit |>> fun chars -> String(Array.ofList chars)
        let sign = opt (pchar '-')
        
        sign >>= fun signOpt ->
        digits >>= fun numStr ->
        let fullStr = match signOpt with Some _ -> "-" + numStr | None -> numStr
        match Int32.TryParse(fullStr) with
        | (true, value) -> succeed (TOMLInteger value)
        | (false, _) -> compilerFail (ParseError(
            { Line = 0; Column = 0; File = ""; Offset = 0 },
            sprintf "Invalid integer: %s" fullStr,
            ["TOML integer parsing"]))
        |> withErrorContext "TOML integer"
    
    /// Parses TOML float literals
    let tomlFloat : Parser<TOMLValue, TOMLParsingState> =
        let digits = many1 digit |>> fun chars -> String(Array.ofList chars)
        let sign = opt (pchar '-')
        
        sign >>= fun signOpt ->
        digits >>= fun intPart ->
        pchar '.' >>= fun _ ->
        digits >>= fun fracPart ->
        let fullStr = 
            match signOpt with 
            | Some _ -> sprintf "-%s.%s" intPart fracPart
            | None -> sprintf "%s.%s" intPart fracPart
        match Double.TryParse(fullStr) with
        | (true, value) -> succeed (TOMLFloat value)
        | (false, _) -> compilerFail (ParseError(
            { Line = 0; Column = 0; File = ""; Offset = 0 },
            sprintf "Invalid float: %s" fullStr,
            ["TOML float parsing"]))
        |> withErrorContext "TOML float"
    
    /// Parses TOML boolean literals
    let tomlBoolean : Parser<TOMLValue, TOMLParsingState> =
        (pstring "true" |>> fun _ -> TOMLBoolean true) <|>
        (pstring "false" |>> fun _ -> TOMLBoolean false)
        |> withErrorContext "TOML boolean"
    
    /// Parses TOML arrays
    let tomlArray : Parser<TOMLValue, TOMLParsingState> =
        let rec parseTomlValue() = 
            tomlString <|> tomlFloat <|> tomlInteger <|> tomlBoolean <|> tomlArray
        
        between (pchar '[') (pchar ']') (sepBy (parseTomlValue()) (pchar ',')) |>>
        TOMLArray
        |> withErrorContext "TOML array"
    
    /// Parses any TOML value
    let tomlValue : Parser<TOMLValue, TOMLParsingState> =
        tomlString <|> tomlFloat <|> tomlInteger <|> tomlBoolean <|> tomlArray
        |> withErrorContext "TOML value"
    
    /// Parses TOML key names
    let tomlKey : Parser<string, TOMLParsingState> =
        let bareKey = many1 (alphaNum <|> pchar '_' <|> pchar '-') |>> fun chars -> String(Array.ofList chars)
        let quotedKey = between (pchar '"') (pchar '"') (many (satisfy (fun c -> c <> '"'))) |>> fun chars -> String(Array.ofList chars)
        
        bareKey <|> quotedKey
        |> withErrorContext "TOML key"
    
    /// Parses TOML key-value pairs
    let tomlKeyValue : Parser<string * TOMLValue, TOMLParsingState> =
        tomlKey >>= fun key ->
        tomlWs >>= fun _ ->
        pchar '=' >>= fun _ ->
        tomlWs >>= fun _ ->
        tomlValue >>= fun value ->
        succeed (key, value)
        |> withErrorContext "TOML key-value pair"
    
    /// Parses TOML section headers
    let tomlSection : Parser<string list, TOMLParsingState> =
        between (pchar '[') (pchar ']') (sepBy tomlKey (pchar '.'))
        |> withErrorContext "TOML section header"
    
    /// Parses complete TOML document
    let tomlDocument : Parser<Map<string list, Map<string, TOMLValue>>, TOMLParsingState> =
        let parseSection (currentSection: string list) : Parser<(string list * Map<string, TOMLValue>), TOMLParsingState> =
            many (tomlWs >>= fun _ -> tomlKeyValue .>> tomlWs) >>= fun pairs ->
            succeed (currentSection, Map.ofList pairs)
        
        let parseSectionWithHeader : Parser<(string list * Map<string, TOMLValue>), TOMLParsingState> =
            tomlSection >>= fun sectionName ->
            tomlWs >>= fun _ ->
            parseSection sectionName
        
        let parseRootSection : Parser<(string list * Map<string, TOMLValue>), TOMLParsingState> =
            parseSection []
        
        tomlWs >>= fun _ ->
        opt parseRootSection >>= fun rootOpt ->
        many (tomlWs >>= fun _ -> parseSectionWithHeader) >>= fun sections ->
        
        let allSections = 
            match rootOpt with
            | Some root -> root :: sections
            | None -> sections
        
        succeed (Map.ofList allSections)
        |> withErrorContext "TOML document"

/// Configuration value extraction using XParsec patterns
module ConfigExtraction =
    
    /// Extracts string value from TOML data
    let extractString (sections: Map<string list, Map<string, TOMLValue>>) (sectionPath: string list) (key: string) (defaultValue: string) : CompilerResult<string> =
        match Map.tryFind sectionPath sections with
        | Some sectionData ->
            match Map.tryFind key sectionData with
            | Some (TOMLString value) -> Success value
            | Some other -> 
                CompilerFailure [ParseError(
                    { Line = 0; Column = 0; File = ""; Offset = 0 },
                    sprintf "Expected string for key '%s', got %A" key other,
                    ["config extraction"])]
            | None -> Success defaultValue
        | None -> Success defaultValue
    
    /// Extracts boolean value from TOML data
    let extractBoolean (sections: Map<string list, Map<string, TOMLValue>>) (sectionPath: string list) (key: string) (defaultValue: bool) : CompilerResult<bool> =
        match Map.tryFind sectionPath sections with
        | Some sectionData ->
            match Map.tryFind key sectionData with
            | Some (TOMLBoolean value) -> Success value
            | Some (TOMLString "true") -> Success true
            | Some (TOMLString "false") -> Success false
            | Some other ->
                CompilerFailure [ParseError(
                    { Line = 0; Column = 0; File = ""; Offset = 0 },
                    sprintf "Expected boolean for key '%s', got %A" key other,
                    ["config extraction"])]
            | None -> Success defaultValue
        | None -> Success defaultValue
    
    /// Extracts integer value from TOML data
    let extractInteger (sections: Map<string list, Map<string, TOMLValue>>) (sectionPath: string list) (key: string) (defaultValue: int option) : CompilerResult<int option> =
        match Map.tryFind sectionPath sections with
        | Some sectionData ->
            match Map.tryFind key sectionData with
            | Some (TOMLInteger value) -> Success (Some value)
            | Some (TOMLString str) ->
                match Int32.TryParse(str) with
                | (true, value) -> Success (Some value)
                | (false, _) -> 
                    CompilerFailure [ParseError(
                        { Line = 0; Column = 0; File = ""; Offset = 0 },
                        sprintf "Cannot convert '%s' to integer for key '%s'" str key,
                        ["config extraction"])]
            | Some other ->
                CompilerFailure [ParseError(
                    { Line = 0; Column = 0; File = ""; Offset = 0 },
                    sprintf "Expected integer for key '%s', got %A" key other,
                    ["config extraction"])]
            | None -> Success defaultValue
        | None -> Success defaultValue
    
    /// Extracts dependency configurations
    let extractDependencies (sections: Map<string list, Map<string, TOMLValue>>) : CompilerResult<DependencyBinding list> =
        let dependencySections = 
            sections 
            |> Map.toList
            |> List.filter (fun (path, _) -> 
                path.Length >= 2 && path.[0] = "dependencies")
        
        let extractDependency (path: string list, data: Map<string, TOMLValue>) : CompilerResult<DependencyBinding> =
            if path.Length >= 2 then
                let depName = path.[1]
                extractString sections path "version" "0.1.0" >>= fun version ->
                extractString sections path "binding" "static" >>= fun binding ->
                extractBoolean sections path "selective_linking" true >>= fun selectiveLinking ->
                Success {
                    Name = depName
                    Version = version
                    Binding = binding
                    SelectiveLinking = selectiveLinking
                }
            else
                CompilerFailure [ParseError(
                    { Line = 0; Column = 0; File = ""; Offset = 0 },
                    "Invalid dependency section path",
                    ["dependency extraction"])]
        
        dependencySections
        |> List.map extractDependency
        |> List.fold (fun acc result ->
            match acc, result with
            | Success deps, Success dep -> Success (dep :: deps)
            | CompilerFailure errors, Success _ -> CompilerFailure errors
            | Success _, CompilerFailure errors -> CompilerFailure errors
            | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
        ) (Success [])
        |>> List.rev

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

/// Parses complete Firefly configuration using XParsec - NO EXTERNAL DEPENDENCIES
let parseConfigFile (filePath: string) : CompilerResult<FireflyConfig> =
    if not (File.Exists(filePath)) then
        Success defaultConfig
    else
        try
            let tomlString = File.ReadAllText(filePath)
            let initialState = {
                CurrentSection = []
                ConfigData = Map.empty
                ErrorContext = []
            }
            
            match TOMLParsers.tomlDocument tomlString initialState with
            | Reply(Ok sections, _) ->
                // Extract package information
                ConfigExtraction.extractString sections ["package"] "name" defaultConfig.PackageName >>= fun packageName ->
                ConfigExtraction.extractString sections ["package"] "version" defaultConfig.Version >>= fun version ->
                
                // Extract compilation configuration
                ConfigExtraction.extractBoolean sections ["compilation"] "require_static_memory" defaultConfig.Compilation.RequireStaticMemory >>= fun requireStaticMemory ->
                ConfigExtraction.extractBoolean sections ["compilation"] "eliminate_closures" defaultConfig.Compilation.EliminateClosures >>= fun eliminateClosures ->
                ConfigExtraction.extractString sections ["compilation"] "optimize" defaultConfig.Compilation.OptimizationLevel >>= fun optimizationLevel ->
                ConfigExtraction.extractInteger sections ["compilation"] "max_stack_size" defaultConfig.Compilation.StackLimit >>= fun stackLimit ->
                ConfigExtraction.extractBoolean sections ["compilation"] "keep_intermediates" defaultConfig.Compilation.KeepIntermediates >>= fun keepIntermediates ->
                
                // Extract LTO setting
                ConfigExtraction.extractString sections ["compilation"] "lto" "false" >>= fun ltoValue ->
                let useLTO = ltoValue.ToLowerInvariant() = "full"
                
                // Extract dependencies
                ConfigExtraction.extractDependencies sections >>= fun dependencies ->
                
                let compilation = {
                    RequireStaticMemory = requireStaticMemory
                    EliminateClosures = eliminateClosures
                    OptimizationLevel = optimizationLevel
                    StackLimit = stackLimit
                    KeepIntermediates = keepIntermediates
                    UseLTO = useLTO
                }
                
                Success {
                    PackageName = packageName
                    Version = version
                    Dependencies = dependencies
                    Compilation = compilation
                }
            
            | Reply(Error, errorMsg) ->
                CompilerFailure [ParseError(
                    { Line = 0; Column = 0; File = filePath; Offset = 0 },
                    sprintf "TOML parsing failed: %s" errorMsg,
                    ["configuration parsing"])]
        
        with
        | ex ->
            CompilerFailure [ParseError(
                { Line = 0; Column = 0; File = filePath; Offset = 0 },
                sprintf "Error reading configuration file: %s" ex.Message,
                ["file reading"])]

/// Validates configuration for consistency and correctness
let validateConfig (config: FireflyConfig) : CompilerResult<FireflyConfig> =
    // Validate package name
    if String.IsNullOrWhiteSpace(config.PackageName) then
        CompilerFailure [ParseError(
            { Line = 0; Column = 0; File = ""; Offset = 0 },
            "Package name cannot be empty",
            ["config validation"])]
    
    // Validate version format (basic check)
    elif not (config.Version.Contains(".")) then
        CompilerFailure [ParseError(
            { Line = 0; Column = 0; File = ""; Offset = 0 },
            "Version must contain at least one dot (e.g., '1.0.0')",
            ["config validation"])]
    
    // Validate optimization level
    elif not (List.contains (config.Compilation.OptimizationLevel.ToLowerInvariant()) 
                ["none"; "less"; "default"; "aggressive"; "size"; "sizemin"]) then
        CompilerFailure [ParseError(
            { Line = 0; Column = 0; File = ""; Offset = 0 },
            sprintf "Invalid optimization level: %s" config.Compilation.OptimizationLevel,
            ["config validation"])]
    
    // Validate stack limit
    elif config.Compilation.StackLimit.IsSome && config.Compilation.StackLimit.Value <= 0 then
        CompilerFailure [ParseError(
            { Line = 0; Column = 0; File = ""; Offset = 0 },
            "Stack limit must be positive",
            ["config validation"])]
    
    else
        Success config

/// Entry point for configuration parsing with validation - NO FALLBACKS
let loadAndValidateConfig (filePath: string) : CompilerResult<FireflyConfig> =
    parseConfigFile filePath >>= validateConfig

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
        CompilerFailure [CompilerError(
            "config generation", 
            sprintf "Failed to write default config to %s" filePath,
            Some ex.Message)]