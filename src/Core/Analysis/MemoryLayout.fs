module Core.Analysis.MemoryLayout

open FSharp.Compiler.Symbols
open Core.Analysis.CouplingCohesion
open Core.Templates.TemplateTypes
open Core.FCS.Helpers  // For getDeclaringEntity

/// Memory region assignment
type MemoryRegion = {
    Name: string
    BaseAddress: uint64 option
    Size: int
    Attributes: MemoryAttribute list
    AssignedUnits: SemanticUnit list
}

and MemoryAttribute =
    | Fast          // TCM, cache-optimized
    | Secure        // TrustZone, protected
    | Shared        // Multi-accessor
    | Persistent    // Non-volatile
    | DMA          // DMA-capable
    | Stack        // Stack-allocated

/// Memory layout strategy derived from analysis
type LayoutStrategy = {
    Regions: MemoryRegion list
    Allocations: Map<string, AllocationInfo>  // Changed from Map<FSharpSymbol, AllocationInfo>
    CrossRegionLinks: CrossRegionLink list
}

and AllocationInfo = {
    Region: string
    Offset: int option
    Size: int
    Alignment: int
}

and CrossRegionLink = {
    From: string  // Region name
    To: string    // Region name
    LinkType: LinkType
}

and LinkType =
    | ZeroCopy      // Shared memory
    | Copy          // Explicit copy needed
    | DMATransfer   // DMA transfer (renamed to avoid confusion)

/// Helper to get symbol identifier
let private getSymbolId (symbol: FSharpSymbol) = symbol.FullName

/// Apply platform template to layout hints
let applyPlatformConstraints (hints: MemoryLayoutHint list) (template: PlatformTemplate) =
    let availableRegions = 
        template.MemoryRegions
        |> List.map (fun r -> {
            Name = r.Name
            BaseAddress = r.BaseAddress
            Size = r.Size
            Attributes = 
                r.Attributes |> List.choose (function
                    | "fast" -> Some Fast
                    | "secure" -> Some Secure
                    | "shared" -> Some Shared
                    | "persistent" -> Some Persistent
                    | "dma" -> Some DMA
                    | "stack" -> Some Stack
                    | _ -> None
                )
            AssignedUnits = []
        })
    
    // Map hints to available regions
    hints |> List.map (fun hint ->
        match hint with
        | Contiguous units ->
            // Find best region for contiguous allocation
            let bestRegion = 
                availableRegions
                |> List.tryFind (fun r -> 
                    r.Attributes |> List.contains Fast
                )
                |> Option.defaultValue (List.head availableRegions)
            
            { bestRegion with AssignedUnits = units }
        
        | Isolated unit ->
            // Prefer secure/isolated regions
            let bestRegion = 
                availableRegions
                |> List.tryFind (fun r -> 
                    r.Attributes |> List.contains Secure
                )
                |> Option.defaultValue (List.head availableRegions)
            
            { bestRegion with AssignedUnits = [unit] }
        
        | SharedRegion (units, pattern) ->
            // Find region suitable for access pattern
            let bestRegion = 
                match pattern with
                | Sequential | Streaming ->
                    availableRegions |> List.tryFind (fun r -> 
                        r.Attributes |> List.contains DMA
                    )
                | Concurrent ->
                    availableRegions |> List.tryFind (fun r -> 
                        r.Attributes |> List.contains Shared
                    )
                | Random ->
                    availableRegions |> List.tryHead
                |> Option.defaultValue (List.head availableRegions)
            
            { bestRegion with AssignedUnits = units }
        
        | Tiered (hot, cold) ->
            // Hot in fast memory, cold in regular memory
            let fastRegion = 
                availableRegions |> List.find (fun r -> 
                    r.Attributes |> List.contains Fast
                )
            let normalRegion = 
                availableRegions |> List.find (fun r -> 
                    not (r.Attributes |> List.contains Fast)
                )
            
            { fastRegion with AssignedUnits = hot }
    )

/// Type to hold symbol and allocation info temporarily
type private SymbolAllocation = {
    SymbolId: string
    Symbol: FSharpSymbol
    Allocation: AllocationInfo
}

/// Calculate memory layout from components and platform
let calculateMemoryLayout 
    (codeComponents: CodeComponent list) 
    (couplings: Coupling list)
    (template: PlatformTemplate) =
    
    // Generate layout hints from coupling/cohesion
    let hints = generateMemoryLayoutHints codeComponents couplings
    
    // Apply platform constraints
    let regions = applyPlatformConstraints hints template
    
    // Build allocation map
    let symbolAllocations = 
        regions
        |> List.collect (fun region ->
            region.AssignedUnits
            |> List.collect (fun unit ->
                match unit with
                | Module entity ->
                    [{
                        SymbolId = getSymbolId (entity :> FSharpSymbol)
                        Symbol = entity :> FSharpSymbol
                        Allocation = {
                            Region = region.Name
                            Offset = None  // Computed later
                            Size = 1024    // Estimate
                            Alignment = 8
                        }
                    }]
                | FunctionGroup functions ->
                    functions |> List.map (fun f ->
                        {
                            SymbolId = getSymbolId (f :> FSharpSymbol)
                            Symbol = f :> FSharpSymbol
                            Allocation = {
                                Region = region.Name
                                Offset = None
                                Size = 256
                                Alignment = 16
                            }
                        }
                    )
                | _ -> []
            )
        )
    
    let allocations = 
        symbolAllocations
        |> List.map (fun sa -> sa.SymbolId, sa.Allocation)
        |> Map.ofList
    
    // Identify cross-region links
    let crossRegionLinks = 
        couplings
        |> List.choose (fun coupling ->
            // Find regions for coupled units
            let fromRegion = 
                regions |> List.tryFind (fun r -> 
                    r.AssignedUnits |> List.contains coupling.From
                )
            let toRegion = 
                regions |> List.tryFind (fun r -> 
                    r.AssignedUnits |> List.contains coupling.To
                )
            
            match fromRegion, toRegion with
            | Some fr, Some tr when fr.Name <> tr.Name ->
                Some {
                    From = fr.Name
                    To = tr.Name
                    LinkType = 
                        if coupling.Strength > 0.8 then ZeroCopy
                        elif fr.Attributes |> List.contains DMA then DMATransfer
                        else Copy
                }
            | _ -> None
        )
        |> List.distinct
    
    {
        Regions = regions
        Allocations = allocations
        CrossRegionLinks = crossRegionLinks
    }

/// Memory safety validation
type MemorySafetyViolation = {
    Description: string
    ViolationType: ViolationType
    Region: string option
}

and ViolationType =
    | BoundsViolation
    | RegionViolation
    | AlignmentViolation
    | CapacityViolation

/// Validate memory layout for safety
let validateMemorySafety (layout: LayoutStrategy) (template: PlatformTemplate) =
    let violations = ResizeArray<MemorySafetyViolation>()
    
    // Check region capacity
    for region in layout.Regions do
        let totalSize = 
            layout.Allocations
            |> Map.toList
            |> List.filter (fun (_, alloc) -> alloc.Region = region.Name)
            |> List.sumBy (fun (_, alloc) -> alloc.Size)
        
        if totalSize > region.Size then
            violations.Add {
                Description = $"Region {region.Name} exceeds capacity: {totalSize}/{region.Size} bytes"
                ViolationType = CapacityViolation
                Region = Some region.Name
            }
    
    // Check cross-region access safety
    for link in layout.CrossRegionLinks do
        let fromRegion = layout.Regions |> List.find (fun r -> r.Name = link.From)
        let toRegion = layout.Regions |> List.find (fun r -> r.Name = link.To)
        
        // Validate security boundaries
        if fromRegion.Attributes |> List.contains Secure &&
           not (toRegion.Attributes |> List.contains Secure) &&
           link.LinkType = ZeroCopy then
            violations.Add {
                Description = $"Zero-copy link from secure region {link.From} to non-secure {link.To}"
                ViolationType = RegionViolation
                Region = Some link.From
            }
    
    List.ofSeq violations

/// Generate memory layout report
let generateLayoutReport (layout: LayoutStrategy) (violations: MemorySafetyViolation list) =
    {|
        Regions = 
            layout.Regions
            |> List.map (fun r ->
                let used = 
                    layout.Allocations
                    |> Map.toList
                    |> List.filter (fun (_, alloc) -> alloc.Region = r.Name)
                    |> List.sumBy (fun (_, alloc) -> alloc.Size)
                
                {|
                    Name = r.Name
                    Size = r.Size
                    Used = used
                    Utilization = float used / float r.Size * 100.0
                    Attributes = r.Attributes |> List.map string
                    Units = r.AssignedUnits.Length
                |}
            )
        
        CrossRegionLinks = 
            layout.CrossRegionLinks
            |> List.map (fun link ->
                {|
                    From = link.From
                    To = link.To
                    Type = string link.LinkType
                |}
            )
        
        Violations = 
            violations
            |> List.map (fun v ->
                {|
                    Type = string v.ViolationType
                    Description = v.Description
                    Region = v.Region |> Option.defaultValue "N/A"
                |}
            )
        
        TotalAllocations = layout.Allocations.Count
        TotalSize = 
            layout.Allocations
            |> Map.toList
            |> List.sumBy (fun (_, alloc) -> alloc.Size)
    |}