module Core.Templates.TemplateTypes

/// Platform template defining memory constraints and capabilities
type PlatformTemplate = {
    Name: string
    Version: string
    Platform: PlatformInfo
    MemoryRegions: MemoryRegionDef list
    Capabilities: Capability list
    Profiles: Profile list
}

and PlatformInfo = {
    Family: string          // "STM32", "Apple-Silicon", "x86-64"
    Architecture: string    // "ARMv8-M", "ARM64", "x86-64"
    Variant: string option  // "STM32L5", "M2", "Zen3"
}

and MemoryRegionDef = {
    Name: string
    BaseAddress: uint64 option
    Size: int
    Attributes: string list
    Access: AccessPermission
}

and AccessPermission =
    | ReadWrite
    | ReadOnly
    | ExecuteOnly
    | NoAccess

and Capability =
    | TrustZone
    | CryptoAcceleration
    | VectorInstructions of width: int
    | DMA of channels: int
    | CXL of version: string
    | ResizableBAR
    | NUMA of nodes: int

and Profile = {
    Name: string
    Description: string
    Requirements: string list
    AllocationStrategy: AllocationStrategy
    Optimizations: Optimization list
}

and AllocationStrategy =
    | StaticPools
    | DynamicHeap
    | StackOnly
    | Hybrid

and Optimization =
    | PreferRegion of region: string
    | AvoidRegion of region: string
    | AlignTo of bytes: int
    | PoolSize of bytes: int

/// Template selection criteria
type TemplateCriteria = {
    Platform: string option
    MinMemory: int option
    RequiredCapabilities: Capability list
    Profile: string option
}

/// Template validation result
type ValidationResult =
    | Valid
    | Invalid of errors: ValidationError list

and ValidationError = {
    Field: string
    Message: string
}

/// Common platform templates
module CommonTemplates =
    let stm32l5 = {
        Name = "STM32L5"
        Version = "1.0"
        Platform = {
            Family = "STM32"
            Architecture = "ARMv8-M"
            Variant = Some "STM32L5"
        }
        MemoryRegions = [
            {
                Name = "TCM"
                BaseAddress = Some 0x20000000UL
                Size = 64 * 1024
                Attributes = ["fast"; "secure"]
                Access = ReadWrite
            }
            {
                Name = "SRAM1"
                BaseAddress = Some 0x20010000UL
                Size = 192 * 1024
                Attributes = ["cached"]
                Access = ReadWrite
            }
            {
                Name = "Flash"
                BaseAddress = Some 0x08000000UL
                Size = 512 * 1024
                Attributes = ["persistent"; "secure"]
                Access = ExecuteOnly
            }
        ]
        Capabilities = [
            TrustZone
            CryptoAcceleration
            DMA 8
        ]
        Profiles = [
            {
                Name = "secure_iot"
                Description = "IoT device with security focus"
                Requirements = ["trustzone"; "crypto"]
                AllocationStrategy = StaticPools
                Optimizations = [
                    PreferRegion "TCM"
                    AlignTo 8
                ]
            }
        ]
    }
    
    let appleM2 = {
        Name = "Apple-M2"
        Version = "1.0"
        Platform = {
            Family = "Apple-Silicon"
            Architecture = "ARM64"
            Variant = Some "M2"
        }
        MemoryRegions = [
            {
                Name = "Unified"
                BaseAddress = None  // Dynamic
                Size = 8 * 1024 * 1024 * 1024  // 8GB minimum
                Attributes = ["unified"; "cached"]
                Access = ReadWrite
            }
            {
                Name = "GPU-Shared"
                BaseAddress = None
                Size = 2 * 1024 * 1024 * 1024  // 2GB
                Attributes = ["gpu"; "shared"]
                Access = ReadWrite
            }
        ]
        Capabilities = [
            VectorInstructions 512
            ResizableBAR
            NUMA 1
        ]
        Profiles = [
            {
                Name = "high_performance"
                Description = "Maximum performance configuration"
                Requirements = []
                AllocationStrategy = DynamicHeap
                Optimizations = [
                    AlignTo 64  // Cache line
                    PoolSize (1024 * 1024 * 64)  // 64MB pools
                ]
            }
        ]
    }
    
    let x86Server = {
        Name = "x86-Server"
        Version = "1.0"
        Platform = {
            Family = "x86-64"
            Architecture = "x86-64"
            Variant = None
        }
        MemoryRegions = [
            {
                Name = "NUMA0"
                BaseAddress = None
                Size = 32 * 1024 * 1024 * 1024  // 32GB
                Attributes = ["numa"; "local"]
                Access = ReadWrite
            }
            {
                Name = "NUMA1"
                BaseAddress = None
                Size = 32 * 1024 * 1024 * 1024  // 32GB
                Attributes = ["numa"; "remote"]
                Access = ReadWrite
            }
            {
                Name = "CXL"
                BaseAddress = None
                Size = 64 * 1024 * 1024 * 1024  // 64GB
                Attributes = ["cxl"; "coherent"]
                Access = ReadWrite
            }
        ]
        Capabilities = [
            CXL "3.0"
            NUMA 2
            VectorInstructions 512
            ResizableBAR
        ]
        Profiles = [
            {
                Name = "data_analytics"
                Description = "Large-scale data processing"
                Requirements = ["numa"; "cxl"]
                AllocationStrategy = Hybrid
                Optimizations = [
                    PreferRegion "CXL"  // For large datasets
                    AlignTo 4096  // Page size
                ]
            }
        ]
    }

/// Template registry
type TemplateRegistry = {
    Templates: Map<string, PlatformTemplate>
    CustomTemplates: Map<string, PlatformTemplate>
}

/// Create empty registry
let emptyRegistry = {
    Templates = 
        [
            "stm32l5", CommonTemplates.stm32l5
            "apple-m2", CommonTemplates.appleM2
            "x86-server", CommonTemplates.x86Server
        ]
        |> Map.ofList
    CustomTemplates = Map.empty
}

/// Validate template
let validateTemplate (template: PlatformTemplate) : ValidationResult =
    let errors = ResizeArray<ValidationError>()
    
    // Check memory regions
    for region in template.MemoryRegions do
        if region.Size <= 0 then
            errors.Add {
                Field = $"MemoryRegion.{region.Name}.Size"
                Message = "Size must be positive"
            }
        
        // Check for overlapping regions if addresses specified
        match region.BaseAddress with
        | Some addr ->
            for other in template.MemoryRegions do
                if other.Name <> region.Name then
                    match other.BaseAddress with
                    | Some otherAddr ->
                        let regionEnd = addr + uint64 region.Size
                        let otherEnd = otherAddr + uint64 other.Size
                        if addr < otherEnd && otherAddr < regionEnd then
                            errors.Add {
                                Field = $"MemoryRegion.{region.Name}"
                                Message = $"Overlaps with region {other.Name}"
                            }
                    | None -> ()
        | None -> ()
    
    // Validate profiles
    for profile in template.Profiles do
        for req in profile.Requirements do
            let hasCapability = 
                template.Capabilities 
                |> List.exists (fun cap ->
                    match cap with
                    | TrustZone -> req = "trustzone"
                    | CryptoAcceleration -> req = "crypto"
                    | CXL _ -> req = "cxl"
                    | NUMA _ -> req = "numa"
                    | _ -> false
                )
            
            if not hasCapability then
                errors.Add {
                    Field = $"Profile.{profile.Name}.Requirements"
                    Message = $"Required capability '{req}' not available"
                }
    
    if errors.Count = 0 then Valid
    else Invalid (List.ofSeq errors)

/// Select template based on criteria
let selectTemplate (criteria: TemplateCriteria) (registry: TemplateRegistry) =
    let allTemplates = 
        Map.toList registry.Templates @ Map.toList registry.CustomTemplates
        |> List.map snd
    
    // Filter by platform
    let filtered1 = 
        match criteria.Platform with
        | Some p -> allTemplates |> List.filter (fun t -> t.Name = p)
        | None -> allTemplates
    
    // Filter by memory requirements
    let filtered2 = 
        match criteria.MinMemory with
        | Some minMem ->
            filtered1 |> List.filter (fun t ->
                let totalMem = t.MemoryRegions |> List.sumBy (fun r -> r.Size)
                totalMem >= minMem
            )
        | None -> filtered1
    
    // Filter by capabilities
    let filtered3 = 
        filtered2 |> List.filter (fun t ->
            criteria.RequiredCapabilities |> List.forall (fun reqCap ->
                t.Capabilities |> List.contains reqCap
            )
        )
    
    // Filter by profile
    let filtered4 = 
        match criteria.Profile with
        | Some profileName ->
            filtered3 |> List.filter (fun t ->
                t.Profiles |> List.exists (fun p -> p.Name = profileName)
            )
        | None -> filtered3
    
    List.tryHead filtered4