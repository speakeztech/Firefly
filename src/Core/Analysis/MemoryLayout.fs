module Core.Analysis.MemoryLayout

open FSharp.Compiler.Symbols
open Core.Analysis.CouplingCohesion
open Core.Templates.TemplateTypes

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
    Allocations: Map<FSharpSymbol, AllocationInfo>
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
    | DMA           // DMA transfer

/// Apply platform template to layout hints
let applyPlatformConstraints (hints: MemoryLayoutHint list) (template: PlatformTemplate) =
    let availableRegions = 
        template.MemoryRegions
        |> List.map (fun r -> {
            Name = r.Name
            BaseAddress = r.BaseAddress
            Size = r.Size
            Attributes = 
                r.Attributes |> List.map (function
                    | "fast" -> Fast
                    | "secure" -> Secure
                    | "shared" -> Shared
                    | "persistent" -> Persistent
                    | "dma" -> DMA
                    | _ -> Stack
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
    let allocations = 
        regions
        |> List.collect (fun region ->
            region.AssignedUnits
            |> List.collect (fun unit ->
                match unit with
                | Module entity ->
                    [{
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
            |> List.map (fun { Symbol = s; Allocation = a } -> s, a)
        )
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
                        elif fr.Attributes |> List.contains DMA then DMA
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
    Symbol: FSharpSymbol
    ViolationType: ViolationType
    Description: string
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
            |> Map.toSeq
            |> Seq.filter (fun (_, alloc) -> alloc.Region = region.Name)
            |> Seq.sumBy (fun (_, alloc) -> alloc.Size)
        
        if totalSize > region.Size then
            violations.Add {
                Symbol = FSharpSymbol()  // Region-level violation
                ViolationType = CapacityViolation
                Description = $"Region {region.Name} exceeds capacity: {totalSize}/{region.Size} bytes"
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
                Symbol = FSharpSymbol()  // Link-level violation
                ViolationType = RegionViolation
                Description = $"Zero-copy link from secure region {link.From} to non-secure {link.To}"
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
                    |> Map.toSeq
                    |> Seq.filter (fun (_, alloc) -> alloc.Region = r.Name)
                    |> Seq.sumBy (fun (_, alloc) -> alloc.Size)
                
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
        
        SafetyViolations = 
            violations
            |> List.map (fun v ->
                {|
                    Type = string v.ViolationType
                    Description = v.Description
                |}
            )
        
        Statistics = {|
            TotalRegions = layout.Regions.Length
            TotalAllocations = layout.Allocations.Count
            CrossRegionLinks = layout.CrossRegionLinks.Length
            SafetyViolations = violations.Length
        |}
    |}

/// Memory layout optimization suggestions
type OptimizationSuggestion = {
    Description: string
    Impact: OptimizationImpact
    Implementation: string
}

and OptimizationImpact =
    | Performance of percent: float
    | Memory of saved: int
    | Safety of resolved: int

/// Generate optimization suggestions
let generateOptimizations (layout: LayoutStrategy) (codeComponents: CodeComponent list) =
    let suggestions = ResizeArray<OptimizationSuggestion>()
    
    // Check for split components
    for codeComp in codeComponents do
        let regions = 
            codeComp.Units
            |> List.choose (fun unit ->
                layout.Regions 
                |> List.tryFind (fun r -> r.AssignedUnits |> List.contains unit)
                |> Option.map (fun r -> r.Name)
            )
            |> List.distinct
        
        if regions.Length > 1 then
            suggestions.Add {
                Description = $"Component {codeComp.Id} is split across {regions.Length} regions"
                Impact = Performance 15.0
                Implementation = "Consolidate component into single region for better cache locality"
            }
    
    // Check for unnecessary cross-region links
    let unnecessaryLinks = 
        layout.CrossRegionLinks
        |> List.filter (fun link -> link.LinkType = Copy)
    
    if unnecessaryLinks.Length > 0 then
        suggestions.Add {
            Description = $"{unnecessaryLinks.Length} cross-region copies could be eliminated"
            Impact = Performance 25.0
            Implementation = "Reorganize units to minimize cross-region dependencies"
        }
    
    List.ofSeq suggestions