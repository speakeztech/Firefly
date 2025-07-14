1. Fix Firefly's PSG Construction Timing

Restructure PSG construction to work with FCS constraint resolution lifecycle
Maybe defer constraint-dependent analysis until after FCS completes constraint resolution
Work with FCS timing rather than against it

2. Use Alternative FCS APIs

The current approach may be using the wrong FCS APIs
There might be constraint-safe ways to access type information during PSG construction
Leverage symbol use correlation instead of direct constraint resolution

3. Constraint Resolution Integration

Instead of avoiding constraint resolution, properly integrate with it
Let FCS complete its constraint resolution, then extract the results
Two-phase approach: generic PSG first, then constraint-resolved PSG

4. Selective Constraint Annotations

Only the specific 232 failing nodes need attention, not all 698
Keep type inference for 99% of the code
Target only the problematic SRTP patterns

The goal should be zero changes to the Alloy library code while fixing the FCS integration in Firefly itself. F# developers shouldn't have to sacrifice type inference because of compiler implementation details.