"""
Provides TableGen specific instantiation of the LanguageServer class using tblgen-lsp-server.
Contains various configurations and settings specific to TableGen (.td files).
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


class TableGenLanguageServer(SolidLanguageServer):
    """
    Provides TableGen specific instantiation of the LanguageServer class using tblgen-lsp-server.
    Supports .td files with features like code completion, go-to-definition, and diagnostics.
    """

    def __init__(self, config: LanguageServerConfig, repository_root_path: str, solidlsp_settings: SolidLSPSettings):
        """
        Creates a TableGenLanguageServer instance. This class is not meant to be instantiated directly.
        Use LanguageServer.create() instead.
        """
        tblgen_lsp_executable_path = self._setup_runtime_dependencies(config, solidlsp_settings)
        super().__init__(
            config,
            repository_root_path,
            ProcessLaunchInfo(cmd=tblgen_lsp_executable_path, cwd=repository_root_path),
            "tablegen",
            solidlsp_settings,
        )
        self.server_ready = threading.Event()
        self.initialize_searcher_command_available = threading.Event()

    @classmethod
    def _setup_runtime_dependencies(cls, config: LanguageServerConfig, solidlsp_settings: SolidLSPSettings) -> str:
        """
        Setup runtime dependencies for TableGen Language Server and return the command to start the server.
        """
        # Look for system-installed tblgen-lsp-server
        tblgen_lsp_executable_path = shutil.which("tblgen-lsp-server")
        if not tblgen_lsp_executable_path:
            raise FileNotFoundError(
                "tblgen-lsp-server is not installed on your system.\n"
                + "Please install MLIR tools using your system package manager:\n"
                + "  Arch Linux: sudo pacman -S mlir\n"
                + "  Ubuntu/Debian: Build from LLVM source or use LLVM packages\n"
                + "  macOS: brew install llvm\n"
                + "See https://mlir.llvm.org/docs/Tools/MLIRLSP/ for more details."
            )
        log.info(f"Using system-installed tblgen-lsp-server at {tblgen_lsp_executable_path}")
        return tblgen_lsp_executable_path

    @staticmethod
    def _get_initialize_params(repository_absolute_path: str) -> InitializeParams:
        """
        Returns the initialize params for the TableGen Language Server.
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
        Starts the TableGen Language Server, waits for the server to be ready.
        """

        def register_capability_handler(params: dict) -> None:
            assert "registrations" in params
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

        log.info("Starting TableGen server process")
        self.server.start()
        initialize_params = self._get_initialize_params(self.repository_root_path)

        log.info("Sending initialize request from LSP client to LSP server and awaiting response")
        init_response = self.server.send.initialize(initialize_params)
        log.debug(f"Received initialize response from TableGen server: {init_response}")

        # Verify basic capabilities
        assert "capabilities" in init_response

        self.server.notify.initialized({})

        # TableGen LSP server is typically ready immediately
        self.server_ready.set()
        self.completions_available.set()
        log.info("TableGen server initialization complete")
