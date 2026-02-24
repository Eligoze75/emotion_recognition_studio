"""Live webcam capture and real-time emotion recognition."""

# Do not import app here: avoids "found in sys.modules before execution" when
# running python -m src.webcam.app and can reduce segfault risk (heavy imports).
__all__ = ["WebcamStream", "LiveEmotionApp"]
