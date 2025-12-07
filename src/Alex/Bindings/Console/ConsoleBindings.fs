module Alex.Bindings.Console.ConsoleBindings

open Alex.Bindings.BindingTypes

/// Register all console bindings for the current platform
let registerAll () =
    // Register Linux console binding
    BindingRegistry.register (Linux.createBinding ())
