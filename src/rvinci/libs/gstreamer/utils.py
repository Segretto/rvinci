import sys
import platform
# Removed logging import as it was unused or standard logging can be used if needed.
# If we want to use rvinci logger:
# from rvinci.core.logging import get_logger
# log = get_logger(__name__)

# GStreamer imports might fail if not installed, handle gracefully or assume environment is set up
try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import GLib, Gst
except ImportError:
    pass


def is_aarch64():
    return platform.machine() == "aarch64"


def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s\n" % (err, debug))
        loop.quit()
    return True
