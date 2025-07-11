from mlagents_envs.side_channel.side_channel import SideChannel, IncomingMessage, OutgoingMessage
from uuid import UUID

# Use a fixed UUID — must match in Unity!
CUSTOM_CHANNEL_ID = UUID("12345678-1234-5678-1234-567812345678")

class CustomDataChannel(SideChannel):
    def __init__(self):
        super().__init__(CUSTOM_CHANNEL_ID)
        self.last_received = None

    def on_message_received(self, msg: IncomingMessage) -> None:
        self.last_received = msg.read_float32()
        print(f"[Python] Unity sent: {self.last_received}")

    def send_data(self, serve: int, p1: int, p2: int):
        msg = OutgoingMessage()
        msg.write_int32(serve)
        msg.write_int32(p1)
        msg.write_int32(p2)
        self.queue_message_to_send(msg)

# Quit channel
class StringSideChannel(SideChannel):
    def __init__(self):
        super().__init__(UUID("e4d8d14a-66b3-4d58-9a3b-b3c32a6fd11b"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        # This can be left empty unless Unity sends something back
        pass

    def send_quit(self):
        msg = OutgoingMessage()
        msg.write_string("shutdown")
        self.queue_message_to_send(msg)
