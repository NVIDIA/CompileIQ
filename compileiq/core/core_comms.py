import json
import subprocess
import sys
import platform
import os
import signal
from pathlib import Path
from pydantic import TypeAdapter
from typing import List, Dict
import socket
from compileiq.config.const import (
    MAX_BYTES,
    MAX_RETRIES,
    SOCKET_TIMEOUT,
)
from compileiq.core.core_types import (
    ResponseTemplate,
    ParameterSet,
    SingleDNA,
    CompletionMessage,
)


"""
CompileIQ's core evolutionary algorithm is compiled in binary form.
For IPC we leverage socket communication through localhost.
"""


class CoreIPC:
    def __init__(self):
        self.core_process = None

    def __del__(self):
        if hasattr(self, "core_process"):
            self.stop()

    def start(
        self,
        server_socket: socket.socket,
        main_config_filepath: str,
        silent: bool = True,
    ) -> subprocess.Popen:
        """
        Starts a subprocess with the Core. It automatically selects the core binary based
        on your operating system and architecture.
        """

        platform_tuple = (sys.platform, platform.machine().lower())  # OS, aarch

        main_binary = os.path.join("bin", "core") if sys.platform != "win32" else "core.exe"
        core_binary = (
            Path(__file__).parent
            / "executable"
            / sys.platform
            / platform.machine().lower()
            / main_binary
        )

        if not os.path.isfile(core_binary):
            raise RuntimeError(
                "CompileIQ's compiled binaries are not supported for "
                f"your platform {platform_tuple}."
            )

        p_core = subprocess.Popen(
            [core_binary, "-c", main_config_filepath],
            env=self.setup_env(server_socket),
            start_new_session=True,
            stdout=subprocess.DEVNULL if silent else sys.stdout,
            stderr=sys.stderr,
        )

        self.core_process = p_core

        return p_core

    def stop(self):
        """
        Kills core subprocess if it is still running.
        """
        # When core's runtime hangs it will not close with a .terminate()
        if (self.core_process is not None) and (self.core_process.poll() is None):
            # Process will hang forever if users control-c at worker execution,
            if sys.platform != "win32":
                # killpg guarantees it closes alongside any other subprocess
                try:
                    os.killpg(os.getpgid(self.core_process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

            self.core_process.kill()

    def receive_from_core(self, socket: socket.socket) -> ParameterSet | CompletionMessage:
        """
        Receive message from core respecting `MAX_BYTES`.
        Bytes are concatenated is built until we can parse it as an json object
        and validate into one of the expected return classes.
        """
        params = None
        partial_data = b""
        retries = 0
        while params is None:
            # Block waiting for core to send data
            try:
                current_data = socket.recvfrom(MAX_BYTES)[0]
            except TimeoutError as e:
                core_return_code = self.core_process.poll()
                if (core_return_code is None) and (retries < MAX_RETRIES):
                    retries += 1
                    continue
                else:
                    raise RuntimeError(
                        "Something went wrong while communicating with core, "
                        "enable debug to access core logs."
                    ) from e

            # Continously build bytestring until we can parse it as a json object
            partial_data += current_data

            try:
                decoded = partial_data.decode("utf-8")
                params = json.loads(decoded)
            except UnicodeDecodeError as e:
                raise RuntimeError("Received invalid UTF-8 data.") from e
            except json.decoder.JSONDecodeError:
                continue
            except Exception as e:
                raise RuntimeError("Something went wrong when validating current samples.") from e

        # Message can be different depending on the mode/stage we are at
        # TODO: Implement a clever identification of message type
        if "generation_num" in params:
            dna_list = TypeAdapter(List[SingleDNA]).validate_python(params.pop("params"))
            received_msg = ParameterSet(params=dna_list, **params)
        else:
            received_msg = CompletionMessage(**params)

        return received_msg

    def setup_env(self, server_socket: socket.socket) -> Dict:
        """
        Core needs some env vars setup to work.
        """
        my_address = server_socket.getsockname()
        current_env = os.environ.copy()
        current_env["CIQ_HOST"] = my_address[0]
        current_env["CIQ_PORT"] = str(my_address[1])

        return current_env

    def send_to_core(
        self,
        socket: socket.socket,
        data: ResponseTemplate,
    ):
        """
        Sends the entire `data` to core through the `socket`.
        Core expects a valid JSON
        """
        # Core expects data to end in a line break
        json_str = data.model_dump_json() + "\n"
        socket.send(json_str.encode())


def initialize_socket(
    bind_to: tuple[str, int] = ("localhost", 0), timeout=SOCKET_TIMEOUT
) -> socket.socket:
    """
    Initializes the socket for IPC communication with Core
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.settimeout(timeout)
    server_socket.bind(bind_to)
    server_socket.listen(1)  # we only accept a single connection from core

    return server_socket
