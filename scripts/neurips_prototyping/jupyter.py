import asyncio
import logging
import os
import signal
import subprocess
from typing import Optional
from uuid import uuid4

import tornado
from loguru import logger
from tornado.escape import json_decode, json_encode, url_escape
from tornado.httpclient import AsyncHTTPClient, HTTPRequest
from tornado.ioloop import IOLoop, PeriodicCallback
from tornado.websocket import websocket_connect


class JupyterKernelGatewayWrapper:
    def __init__(self, ip_address: str = "localhost", port: str = "8888"):
        self.ip_address = ip_address
        self.port = port
        self.process: Optional[subprocess.Popen] = None

    @property
    def command(self) -> list[str]:
        executable = "jupyter"
        args = [
            "kernelgateway",
            f"--ip={self.ip_address}",
            f"--port={self.port}",
            "--JupyterWebsocketPersonality.list_kernels=True",
        ]
        return [executable] + args

    def __enter__(self):
        self.process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Detach the process group so we can return control to the main
            # Python process that spawned the gateway.
            preexec_fn=os.setpgrp,
        )
        logger.info("Jupyter Kernel Gateway started with PID: {}", self.process.pid)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Terminate the process group and kill the gateway.
        assert self.process is not None, "Gateway process not started or already dead?"
        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        logger.info("Jupyter Kernel Gateway stopped with PID: {}", self.process.pid)


class JupyterKernel:
    def __init__(
        self,
        gateway_ip_addr: str,
        gateway_port: str,
        conv_id: str,
        kernel_language="python",
    ):
        self.base_url: str = f"http://{gateway_ip_addr}:{gateway_port}"
        self.base_websocket_url: str = f"ws://{gateway_ip_addr}:{gateway_port}"
        self.kernel_language = kernel_language
        self.kernel_id: Optional[str] = None
        self.websocket = None
        self.conv_id = conv_id
        logger.info(
            "Jupyter kernel created for conversation {} at {}", conv_id, self.base_url
        )
        self.heartbeat_interval = 10000  # 10 seconds
        self.heartbeat_callback = None

    async def initialize(self):
        await self.execute(r"%colors nocolor")
        # pre-defined tools
        self.tools_to_run = [
            # TODO: You can add code for your pre-defined tools here (need to improve this)
        ]
        for tool in self.tools_to_run:
            # logging.info(f"Tool initialized:\n{tool}")
            await self.execute(tool)

    async def _send_heartbeat(self):
        if not self.websocket:
            return
        try:
            self.websocket.ping()
            # logging.info("Heartbeat sent...")
        except tornado.iostream.StreamClosedError:
            # logging.info("Heartbeat failed, reconnecting...")
            try:
                await self._connect()
            except ConnectionRefusedError:
                logger.info(
                    "ConnectionRefusedError: Failed to reconnect to kernel websocket - Is the kernel still running?"
                )

    async def _connect(self):
        if self.websocket:
            self.websocket.close()
            self.websocket = None

        client = AsyncHTTPClient()
        if not self.kernel_id:
            n_tries = 10
            while n_tries > 0:
                try:
                    response = await client.fetch(
                        "{}/api/kernels".format(self.base_url),
                        method="POST",
                        body=json_encode({"name": self.kernel_language}),
                    )
                    kernel = json_decode(response.body)
                    self.kernel_id = kernel["id"]
                    break
                except Exception:
                    logger.opt(exception=True).warning(
                        "Failed to create kernel, retrying..."
                    )
                    # kernels are not ready yet
                    n_tries -= 1
                    await asyncio.sleep(1)

            if n_tries == 0:
                raise ConnectionRefusedError("Failed to connect to kernel")
            else:
                logger.info(f"Kernel created: {self.kernel_id}")
                # Just to convince MyPy that kernel_id is not None.
                assert self.kernel_id is not None

        websocket_request = HTTPRequest(
            url="{}/api/kernels/{}/channels".format(
                self.base_websocket_url, url_escape(self.kernel_id)
            )
        )
        self.websocket = await websocket_connect(websocket_request)
        logger.info("Connected to kernel websocket")

        # Setup heartbeat
        if self.heartbeat_callback:
            self.heartbeat_callback.stop()
        self.heartbeat_callback = PeriodicCallback(
            self._send_heartbeat, self.heartbeat_interval
        )
        self.heartbeat_callback.start()

    async def execute(self, code, timeout=180):
        if not self.websocket:
            await self._connect()

        # fucai debug code
        # print(f'current code:\n{code}\n')
        msg_id = uuid4().hex
        self.websocket.write_message(  # type: ignore[attr-defined]
            json_encode(
                {
                    "header": {
                        "username": "",
                        "version": "5.0",
                        "session": "",
                        "msg_id": msg_id,
                        "msg_type": "execute_request",
                    },
                    "parent_header": {},
                    "channel": "shell",
                    "content": {
                        "code": code,
                        "silent": False,
                        "store_history": False,
                        "user_expressions": {},
                        "allow_stdin": False,
                    },
                    "metadata": {},
                    "buffers": {},
                }
            )
        )

        outputs = []

        async def wait_for_messages():
            execution_done = False
            while not execution_done:
                msg = await self.websocket.read_message()  # type: ignore[attr-defined]
                # I don't know if we should even handle this case; can it happen?
                # In any case, this makes the type checker happy.
                if not msg:
                    print("Empty message received from kernel")
                    break
                msg = json_decode(msg)
                msg_type = msg["msg_type"]
                parent_msg_id = msg["parent_header"].get("msg_id", None)

                if parent_msg_id != msg_id:
                    continue

                if os.environ.get("DEBUG", False):
                    logging.info(
                        f"MSG TYPE: {msg_type.upper()} DONE:{execution_done}\nCONTENT: {msg['content']}"
                    )

                if msg_type == "error":
                    traceback = "\n".join(msg["content"]["traceback"])
                    outputs.append(traceback)
                    execution_done = True
                elif msg_type == "stream":
                    outputs.append(msg["content"]["text"])
                elif msg_type in ["execute_result", "display_data"]:
                    outputs.append(msg["content"]["data"]["text/plain"])
                    if "image/png" in msg["content"]["data"]:
                        # use markdone to display image (in case of large image)
                        # outputs.append(f"\n<img src=\"data:image/png;base64,{msg['content']['data']['image/png']}\"/>\n")
                        outputs.append(
                            f"![image](data:image/png;base64,{msg['content']['data']['image/png']})"
                        )

                elif msg_type == "execute_reply":
                    execution_done = True
            return execution_done

        async def interrupt_kernel():
            client = AsyncHTTPClient()
            interrupt_response = await client.fetch(
                f"{self.base_url}/api/kernels/{self.kernel_id}/interrupt",
                method="POST",
                body=json_encode({"kernel_id": self.kernel_id}),
            )
            logger.info(f"Kernel interrupted: {interrupt_response}")

        try:
            execution_done = await asyncio.wait_for(wait_for_messages(), timeout)
        except asyncio.TimeoutError:
            await interrupt_kernel()
            return f"[Execution timed out ({timeout} seconds).]"

        if not outputs and execution_done:
            ret = "[Code executed successfully with no output]"
        else:
            ret = "".join(outputs)

        if os.environ.get("DEBUG", False):
            logger.info(f"OUTPUT:\n{ret}")
        return ret

    async def shutdown_async(self):
        if self.kernel_id:
            client = AsyncHTTPClient()
            await client.fetch(
                "{}/api/kernels/{}".format(self.base_url, self.kernel_id),
                method="DELETE",
            )
            self.kernel_id = None
            if self.websocket:
                self.websocket.close()
                self.websocket = None


if __name__ == "__main__":
    with JupyterKernelGatewayWrapper() as gateway:

        async def main():
            # kernel = JupyterKernel(
            #     gateway_ip_addr=gateway.ip_address,
            #     gateway_port=gateway.port,
            #     conv_id="test",
            # )
            kernel = JupyterKernel(
                gateway_ip_addr="172.16.21.9",
                gateway_port="7274",
                conv_id="test",
            )
            await kernel.initialize()
            result = await kernel.execute("a = 1")
            print(result)
            result = await kernel.execute("b=2\nc=a+b\nprint(c)")
            print(result)
            # result = await kernel.execute(
            #     "{k: repr(v) for k,v in locals().items() if not k.startswith('_')}"
            # )
            # print(result)
            await kernel.shutdown_async()

        IOLoop.current().run_sync(main)
        import requests

        print(
            requests.get(
                f"http://{gateway.ip_address}:{gateway.port}/api/kernels"
            ).json()
        )
    print("Kernel closed")
