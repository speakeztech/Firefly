"""
Provides LLVM IR specific instantiation of the LanguageServer class using llvm-ir-lsp.
Contains various configurations and settings specific to LLVM IR (.ll files).

Note: This uses the third-party llvm-ir-lsp from https://github.com/indoorvivants/llvm-ir-lsp
which is experimental and has limited features (document symbols, go-to-definition, hover).
"""

import logging
import os
import pathlib
import shutil
import threading
from typing import Any

from solidlsp.ls import SolidLanguageServer
from solidlsp.ls_config import LanguageServerConfig
from solidlsp.lsp_protocol_handler.lsp_types import InitializeParams
from solidlsp.lsp_protocol_handler.server import ProcessLaunchInfo
from solidlsp.settings import SolidLSPSettings

log = logging.getLogger(__name__)


class LLVMIRLanguageServer(SolidLanguageServer):
    """
    Provides LLVM IR specific instantiation of the LanguageServer class using llvm-ir-lsp.
    Supports .ll files with limited features: document symbols, go-to-definition, hover.

    This uses the experimental third-party server from:
    https://github.com/indoorvivants/llvm-ir-lsp
    """

    def __init__(self, config: LanguageServerConfig, repository_root_path: str, solidlsp_settings: SolidLSPSettings):
        """
        Creates a LLVMIRLanguageServer instance. This class is not meant to be instantiated directly.
        Use LanguageServer.create() instead.
        """
        llvmir_lsp_executable_path = self._setup_runtime_dependencies(config, solidlsp_settings)
        super().__init__(
            config,
            repository_root_path,
            ProcessLaunchInfo(cmd=llvmir_lsp_executable_path, cwd=repository_root_path),
            "llvm_ir",
            solidlsp_settings,
        )
        self.server_ready = threading.Event()
        self.initialize_searcher_command_available = threading.Event()

    @classmethod
    def _setup_runtime_dependencies(cls, config: LanguageServerConfig, solidlsp_settings: SolidLSPSettings) -> str:
        """
        Setup runtime dependencies for LLVM IR Language Server and return the command to start the server.
        """
        # Look for llvm-ir-lsp in common locations
        llvmir_lsp_executable_path = shutil.which("llvm-ir-lsp")

        if not llvmir_lsp_executable_path:
            # Check ~/.local/bin explicitly
            local_bin_path = os.path.expanduser("~/.local/bin/llvm-ir-lsp")
            if os.path.exists(local_bin_path) and os.access(local_bin_path, os.X_OK):
                llvmir_lsp_executable_path = local_bin_path

        if not llvmir_lsp_executable_path:
            raise FileNotFoundError(
                "llvm-ir-lsp is not installed on your system.\n"
                + "Please download it from:\n"
                + "  https://github.com/indoorvivants/llvm-ir-lsp/releases\n"
                + "\nInstallation:\n"
                + "  curl -L -o ~/.local/bin/llvm-ir-lsp \\\n"
                + "    'https://github.com/indoorvivants/llvm-ir-lsp/releases/download/v0.0.3/LLVM_LanguageServer-x86_64-pc-linux'\n"
                + "  chmod +x ~/.local/bin/llvm-ir-lsp\n"
            )
        log.info(f"Using llvm-ir-lsp at {llvmir_lsp_executable_path}")
        return llvmir_lsp_executable_path

    @staticmethod
    def _get_initialize_params(repository_absolute_path: str) -> InitializeParams:
        """
        Returns the initialize params for the LLVM IR Language Server.
        """
        root_uri = pathlib.Path(repository_absolute_path).as_uri()
        initialize_params = {
            "locale": "en",
            "capabilities": {
                "textDocument": {
                    "synchronization": {"didSave": True, "dynamicRegistration": True},
                    "completion": {"dynamicRegistration": True, "completionItem": {"snippetSupport": True}},
                    "definition": {"dynamicRegistration": True},
                    "references": {"dynamicRegistration": True},
                    "documentSymbol": {
                        "dynamicRegistration": True,
                        "hierarchicalDocumentSymbolSupport": True,
                        "symbolKind": {"valueSet": list(range(1, 27))},
                    },
                    "hover": {"dynamicRegistration": True, "contentFormat": ["markdown", "plaintext"]},
                    "signatureHelp": {"dynamicRegistration": True},
                    "codeAction": {"dynamicRegistration": True},
                },
                "workspace": {
                    "workspaceFolders": True,
                    "didChangeConfiguration": {"dynamicRegistration": True},
                    "symbol": {"dynamicRegistration": True},
                },
            },
            "processId": os.getpid(),
            "rootPath": repository_absolute_path,
            "rootUri": root_uri,
            "workspaceFolders": [
                {
                    "uri": root_uri,
                    "name": os.path.basename(repository_absolute_path),
                }
            ],
        }
        return initialize_params  # type: ignore

    def _start_server(self) -> None:
        """
        Starts the LLVM IR Language Server, waits for the server to be ready.
        """

        def register_capability_handler(params: dict) -> None:
            if "registrations" in params:
                for registration in params["registrations"]:
                    if registration["method"] == "workspace/executeCommand":
                        self.initialize_searcher_command_available.set()
            return

        def execute_client_command_handler(params: dict) -> list:
            return []

        def do_nothing(params: Any) -> None:
            return

        def window_log_message(msg: dict) -> None:
            log.info(f"LSP: window/logMessage: {msg}")

        self.server.on_request("client/registerCapability", register_capability_handler)
        self.server.on_notification("window/logMessage", window_log_message)
        self.server.on_request("workspace/executeClientCommand", execute_client_command_handler)
        self.server.on_notification("$/progress", do_nothing)
        self.server.on_notification("textDocument/publishDiagnostics", do_nothing)

        log.info("Starting LLVM IR server process")
        self.server.start()
        initialize_params = self._get_initialize_params(self.repository_root_path)

        log.info("Sending initialize request from LSP client to LSP server and awaiting response")
        init_response = self.server.send.initialize(initialize_params)
        log.debug(f"Received initialize response from LLVM IR server: {init_response}")

        # Verify basic capabilities
        assert "capabilities" in init_response

        self.server.notify.initialized({})

        # LLVM IR LSP server is typically ready immediately
        self.server_ready.set()
        self.completions_available.set()
        log.info("LLVM IR server initialization complete")
