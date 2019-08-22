import asyncio
import http.client
import io
import json
import logging
import os
import socket
import sys
import threading
import time
from base64 import b64encode
from contextlib import contextmanager
from ctypes import *
from enum import Enum
from hashlib import sha1
from urllib.parse import urlparse

import websockets

import gi

gi.require_version("Gst", "1.0")  # isort:skip
gi.require_version("GstRtp", "1.0")  # isort:skip
from gi.repository import GLib, Gst, GstRtp  # isort:skip

Gst.init(sys.argv)  # isort:skip

MACOS = os.uname().sysname == "Darwin"


def get_logger(level=logging.INFO):
    try:
        import daiquiri

        daiquiri.setup(outputs=(daiquiri.output.STDERR,), level=level)

        logger = daiquiri.getLogger("roku")
    except ImportError:
        logger = logging.getLogger("roku")

    if os.getenv("DEBUG"):
        logger.setLevel(logging.DEBUG)

    return logger


LOGGER = get_logger()

try:
    import uvloop
except ImportError:
    LOGGER.warning("Running without uvloop")
else:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    LOGGER.info("Running with uvloop")


### START Writable Gst.Buffer workaround ###


class GstMapInfo(Structure):
    _fields_ = [
        ("memory", c_void_p),  # GstMemory *memory
        ("flags", c_int),  # GstMapFlags flags
        ("data", POINTER(c_byte)),  # guint8 *data
        ("size", c_size_t),  # gsize size
        ("maxsize", c_size_t),  # gsize maxsize
        ("user_data", c_void_p * 4),  # gpointer user_data[4]
        ("_gst_reserved", c_void_p * 4),
    ]


if MACOS:
    libgst = CDLL("libgstreamer-1.0.dylib")
else:
    libgst = CDLL("libgstreamer-1.0.so.0")
GST_MAP_INFO_POINTER = POINTER(GstMapInfo)

# gst_buffer_map
libgst.gst_buffer_map.argtypes = [c_void_p, GST_MAP_INFO_POINTER, c_int]
libgst.gst_buffer_map.restype = c_int

# gst_buffer_unmap
libgst.gst_buffer_unmap.argtypes = [c_void_p, GST_MAP_INFO_POINTER]
libgst.gst_buffer_unmap.restype = None

# gst_mini_object_is_writable
libgst.gst_mini_object_is_writable.argtypes = [c_void_p]
libgst.gst_mini_object_is_writable.restype = c_int


@contextmanager
def map_gst_buffer(pbuffer, flags):
    if pbuffer is None:
        raise TypeError("Cannot pass NULL to map_gst_buffer")

    ptr = hash(pbuffer)
    if flags & Gst.MapFlags.WRITE and libgst.gst_mini_object_is_writable(ptr) == 0:
        raise ValueError("Writable array requested but buffer is not writeable")

    mapping = GstMapInfo()
    success = libgst.gst_buffer_map(ptr, mapping, flags)

    if not success:
        raise RuntimeError("Couldn't map buffer")

    try:
        # Cast POINTER(c_byte) to POINTER to array of c_byte with size mapping.size
        # Returns not pointer but the object to which pointer points
        yield cast(mapping.data, POINTER(c_byte * mapping.size)).contents
    finally:
        libgst.gst_buffer_unmap(ptr, mapping)


### END Writable Gst.Buffer workaround ###

### START SSDP Discovery code ###


class SSDPResponse:
    class _FakeSocket(io.BytesIO):
        def makefile(self, *args, **kw):  # pylint: disable=unused-argument
            return self

    def __init__(self, response):
        r = http.client.HTTPResponse(self._FakeSocket(response))
        r.begin()
        self.location = r.getheader("location")
        self.usn = r.getheader("usn")
        self.st = r.getheader("st")
        self.cache = r.getheader("cache-control").split("=")[1]

    def __repr__(self):
        return "<SSDPResponse({location}, {st}, {usn})>".format(**self.__dict__)


def discover(service, timeout=5, retries=1, mx=3):
    group = ("239.255.255.250", 1900)
    message = "\r\n".join(
        [
            "M-SEARCH * HTTP/1.1",
            "HOST: {0}:{1}",
            'MAN: "ssdp:discover"',
            "ST: {st}",
            "MX: {mx}",
            "",
            "",
        ]
    )
    socket.setdefaulttimeout(timeout)
    responses = {}
    for _ in range(retries):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        message_bytes = message.format(*group, st=service, mx=mx).encode("utf-8")
        sock.sendto(message_bytes, group)
        while True:
            try:
                response = SSDPResponse(sock.recv(1024))
                responses[response.location] = response
            except socket.timeout:
                break
    return list(responses.values())


### END SSDP Discovery code ###


def get_roku_ip_port():
    LOGGER.info("Searching for Roku devices...")

    rokus = discover("roku:ecp")

    if not rokus:
        LOGGER.error("No Roku device found")
        sys.exit(1)

    if len(rokus) > 1:
        LOGGER.warning(
            "Multiple Roku devices found:\n\t%s", "\n\t".join(str(r) for r in rokus)
        )
        LOGGER.info("Using the first one: %s", rokus[0].location)

    LOGGER.info("Found Roku device: %s", rokus[0].location)
    ip, port = urlparse(rokus[0].location).netloc.split(":")
    return ip, int(port)


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    finally:
        s.close()
    return IP


KEY = "95E610D0-7C29-44EF-FB0F-97F1FCE4C297"
RTP_PORT = 6970
RTCP_PORT = 5150
RTCP_APP_PORT = 6971
LATENCY = 97
CLOCK_RATE = 48000

AUDIO_SINK = "osxaudiosink" if MACOS else "alsasink"
SESSION_NUMBER = 0


def char_transform(var1, var2):
    if ord("0") <= var1 <= ord("9"):
        var3 = var1 - 48
    elif ord("A") <= var1 <= ord("F"):
        var3 = var1 - 65 + 10
    else:
        var3 = -1

    if var3 < 0:
        return chr(var1)

    var2 = 15 - var3 + var2 & 15
    if var2 < 10:
        var2 += 48
    else:
        var2 = var2 + 65 - 10

    return chr(var2)


class RokuPacketAppName(Enum):
    AUDIO_VIDEO_SYNC_DELAY = "VDLY"
    CROSS_SYNC_DELAY = "XDLY"
    CLIENT_VERSION = "CVER"
    NEW_CLIENT = "NCLI"


# pylint: disable=too-many-instance-attributes
class RokuAudio:
    AUTH_KEY = "".join(char_transform(ord(c), 9) for c in KEY).encode()

    def __init__(
        self,
        roku_ip=None,
        receiver_ip=None,
        audio_sink=AUDIO_SINK,
        audio_video_sync_delay_ms=200,
        debug=False,
    ):
        self.vdly_sent = False
        self.cver_sent = False
        self.xdly_received = False
        self.ncli_received = False
        self.goodbye_sent = False
        self.should_send_goodbye = False

        self.ssrc_active_at = None
        self.ssrc_active = False

        self.audio_video_sync_delay_ms = audio_video_sync_delay_ms
        self.client_version = b"0002"

        self.request_id = 0
        self.loop = asyncio.get_event_loop()
        self.glib_loop = GLib.MainLoop()
        self.gst_thread = None

        self.pipeline = None
        self.internal_session = None
        self.rtpbin = None
        self.rtpsession = None
        self.rtpsource = None

        self.roku_ip = roku_ip
        self.roku_port = 8060
        self.audio_receiver_ip = receiver_ip
        self.audio_sink = audio_sink

        self.debug = debug
        if debug:
            LOGGER.setLevel(logging.DEBUG)

    @staticmethod
    def auth_key(s):
        return b64encode(sha1(s.encode() + RokuAudio.AUTH_KEY).digest()).decode()

    def auth_response(self, s):
        return {
            "param-microphone-sample-rates": "1600",
            "param-response": RokuAudio.auth_key(s),
            "param-client-friendly-name": "Wireless Speaker",
            "request": "authenticate",
            "param-has-microphone": "false",
        }

    def on_new_ssrc(self, sess, ssrc, udata):  # pylint: disable=unused-argument
        if not self.rtpsource:
            self.rtpsource = ssrc
            self.ssrc_active_at = time.time()
        LOGGER.debug("New SSRC:\n\tSession: %s\n\tSSRC: %s", sess, ssrc)

    def on_app_rtcp(
        self, sess, subtype, ssrc, name, data, udata
    ):  # pylint: disable=unused-argument
        """
            @session: the object which received the signal
            @subtype: The subtype of the packet
            @ssrc: The SSRC/CSRC of the packet
            @name: The name of the packet
            @data: a #GstBuffer with the application-dependant data or %NULL if
            there was no data
        """

        if name == RokuPacketAppName.CROSS_SYNC_DELAY.value:
            success, mapinfo = data.map(Gst.MapFlags.READ)
            if not success:
                LOGGER.warning("Received 'Cross Sync Delay' packet with no data")
                return

            received_data = mapinfo.data
            received_audio_video_sync_delay_ms = (
                int.from_bytes(received_data, "big") // 1000
            )
            LOGGER.info(
                "Received 'Cross Sync Delay' packet:\n\tData: %s\n\tSync Delay: %s ms",
                received_data.hex(),
                received_audio_video_sync_delay_ms,
            )

            data.unmap(mapinfo)
            if received_audio_video_sync_delay_ms == self.audio_video_sync_delay_ms:
                self.xdly_received = True
            else:
                self.vdly_sent = False
                self.audio_video_sync_delay_ms = received_audio_video_sync_delay_ms
            return

        if name == RokuPacketAppName.NEW_CLIENT.value:
            LOGGER.info("Received 'New Client' packet")
            self.ncli_received = True
            return

        LOGGER.debug(
            "App RTCP:\n\tName: %s\n\tData: %s", name, data.hex() if data else data
        )

    def send_audio_video_sync_delay(self, buffer, packet):
        packet.app_set_name(RokuPacketAppName.AUDIO_VIDEO_SYNC_DELAY.value)
        packet.app_set_data_length(1)
        packet.app_set_ssrc(0)

        vdly_data = (self.audio_video_sync_delay_ms * 1000).to_bytes(4, "big")
        LOGGER.info(
            "Sending 'Audio Video Sync Delay' packet:\n\tData: %s\n\tSync Delay: %s ms",
            vdly_data.hex(),
            self.audio_video_sync_delay_ms,
        )

        try:
            with map_gst_buffer(
                buffer, Gst.MapFlags.READ | Gst.MapFlags.WRITE
            ) as mapped:
                mapped[12] = vdly_data[0]
                mapped[13] = vdly_data[1]
                mapped[14] = vdly_data[2]
                mapped[15] = vdly_data[3]
        except Exception as exc:
            LOGGER.exception(exc)
            if self.debug:
                breakpoint()

        self.xdly_received = False
        self.vdly_sent = True

    def send_client_version(self, buffer, packet):
        packet.app_set_name(RokuPacketAppName.CLIENT_VERSION.value)
        packet.app_set_data_length(1)
        packet.app_set_ssrc(0)

        LOGGER.info(
            "Sending 'Client Version' packet:\n\tData: %s\n\tVersion: %s",
            self.client_version.hex(),
            self.client_version,
        )

        try:
            with map_gst_buffer(
                buffer, Gst.MapFlags.READ | Gst.MapFlags.WRITE
            ) as mapped:
                mapped[12] = self.client_version[0]
                mapped[13] = self.client_version[1]
                mapped[14] = self.client_version[2]
                mapped[15] = self.client_version[3]
        except Exception as exc:
            LOGGER.exception(exc)
            if self.debug:
                breakpoint()

        self.cver_sent = True

    def clear_rtcp_packets(self, rtcpbuffer):
        packet = GstRtp.RTCPPacket()
        rtcpbuffer.get_first_packet(packet)

        for _ in range(rtcpbuffer.get_packet_count()):
            LOGGER.debug("Removing packet with type: %s", packet.type)
            packet.remove()

    def on_sending_rtcp(
        self, sess, buffer, early, udata
    ):  # pylint: disable=unused-argument
        rtcpbuffer = GstRtp.RTCPBuffer()
        GstRtp.rtcp_buffer_map(
            buffer, Gst.MapFlags.READ | Gst.MapFlags.WRITE, rtcpbuffer
        )

        packet = GstRtp.RTCPPacket()
        rtcpbuffer.get_first_packet(packet)

        if not self.ssrc_active:
            if not self.ssrc_active_at or time.time() - self.ssrc_active_at < 1:
                self.clear_rtcp_packets(rtcpbuffer)
                return
            self.ssrc_active = True

        if self.should_send_goodbye and not self.goodbye_sent:
            self.clear_rtcp_packets(rtcpbuffer)
            packet = GstRtp.RTCPPacket()
            rtcpbuffer.add_packet(GstRtp.RTCPType.BYE, packet)
            rtcpbuffer.unmap()
            self.goodbye_sent = True
            return

        if rtcpbuffer.get_packet_count() == 2 and packet.type == GstRtp.RTCPType.RR:
            if not self.vdly_sent or not self.cver_sent:
                packet.remove()
                packet = GstRtp.RTCPPacket()
                rtcpbuffer.get_first_packet(packet)
            else:
                packet.rr_set_ssrc(0)
                packet.move_to_next()

            if packet.type == GstRtp.RTCPType.SDES:
                packet.remove()

        rtcp_handshake_done = (
            self.vdly_sent
            and self.cver_sent
            and self.xdly_received
            and self.ncli_received
        )

        if rtcp_handshake_done:
            rtcpbuffer.unmap()
            return

        packet = GstRtp.RTCPPacket()
        rtcpbuffer.add_packet(GstRtp.RTCPType.APP, packet)

        if not self.vdly_sent:
            self.send_audio_video_sync_delay(buffer, packet)
        elif not self.cver_sent and self.xdly_received:
            self.send_client_version(buffer, packet)
        else:
            packet.remove()

        rtcpbuffer.unmap()
        return

    def gst_pipeline(self):
        command = f"""
            rtpbin name=rtpbin rtp-profile=avpf latency={LATENCY} sdes="application/x-rtp-source-sdes,cname=\"\",tool=\"\""
            udpsrc address=0.0.0.0 port={RTP_PORT} caps="application/x-rtp,media=(string)audio,clock-rate=(int){CLOCK_RATE},encoding-name=(string)OPUS" ! rtpbin.recv_rtp_sink_{SESSION_NUMBER}
            rtpbin. ! rtpopusdepay ! queue ! opusdec ! {self.audio_sink}
            udpsrc address=0.0.0.0 port={RTCP_APP_PORT} caps="application/x-rtcp" ! rtpbin.recv_rtcp_sink_{SESSION_NUMBER}
            rtpbin.send_rtcp_src_{SESSION_NUMBER} ! udpsink host="{self.roku_ip}" port={RTCP_PORT} sync=false async=false bind-port={RTP_PORT}
        """
        LOGGER.debug("GStreamer pipeline:\n%s", command)
        self.pipeline = Gst.parse_launch(command)
        self.rtpbin = self.pipeline.children[-1]

        self.rtpsession = self.rtpbin.children[-1]
        self.rtpsession.connect("on-new-sender-ssrc", self.on_new_ssrc, None)

        self.internal_session = self.rtpbin.emit("get-internal-session", SESSION_NUMBER)
        self.internal_session.connect("on-sending-rtcp", self.on_sending_rtcp, None)
        self.internal_session.connect("on-app-rtcp", self.on_app_rtcp, None)

        self.pipeline.set_state(Gst.State.PLAYING)

    async def send(self, ws, obj):
        self.request_id += 1

        resp = json.dumps({**obj, "request-id": str(self.request_id)})
        LOGGER.debug("ECP sending %s", resp)
        await ws.send(resp)

    async def run_ecp(self):
        async with websockets.connect(
            f"ws://{self.roku_ip}:{self.roku_port}/ecp-session",
            origin="Android",
            subprotocols=["ecp-2"],
        ) as ws:
            async for message in ws:
                msg = json.loads(message)
                LOGGER.debug("ECP received %s", msg)

                if "param-challenge" in msg:
                    challenge = msg["param-challenge"]
                    await self.send(ws, self.auth_response(challenge))
                elif (
                    msg.get("response") == "authenticate" and msg.get("status") == "200"
                ):
                    await self.send(
                        ws,
                        {
                            "param-devname": f"{self.audio_receiver_ip}:{RTP_PORT}:{LATENCY}:{CLOCK_RATE // 50}",
                            "param-audio-output": "datagram",
                            "request": "set-audio-output",
                        },
                    )

    async def send_rtcp(self):
        while True:
            if self.internal_session:
                self.internal_session.emit("send-rtcp", SESSION_NUMBER)
            await asyncio.sleep(0.2)

    def run_gst(self):
        self.gst_pipeline()

        try:
            self.glib_loop.run()
        except Exception as exc:
            LOGGER.exception(exc)
            if self.debug:
                breakpoint()

    def send_goodbye(self):
        self.should_send_goodbye = True
        self.internal_session.emit("send-rtcp", SESSION_NUMBER)
        while not self.goodbye_sent:
            time.sleep(0.1)

    def run(self):
        try:
            self.audio_receiver_ip = self.audio_receiver_ip or get_ip()
            if not self.roku_ip:
                self.roku_ip, self.roku_port = get_roku_ip_port()

            self.gst_thread = threading.Thread(target=self.run_gst, daemon=True)
            self.gst_thread.start()

            self.loop.create_task(self.send_rtcp())
            self.loop.run_until_complete(self.run_ecp())
        except KeyboardInterrupt:
            LOGGER.info("Stopping ECP session")
            self.loop.stop()

            LOGGER.info("Sending 'Goodbye' packet")
            self.send_goodbye()

            LOGGER.info("Stopping GStreamer")
            self.glib_loop.quit()

            LOGGER.info("Ciao!")


if __name__ == "__main__":
    import fire

    fire.Fire(RokuAudio)
