from dataclasses import dataclass

@dataclass
class PDW:
    """
    Represents a 5-D Pulse Descriptor Word (PDW).
    Units:
    - toa: Time of Arrival in seconds (s)
    - centre_frequency: Centre frequency in Hertz (Hz)
    - pulse_width: Pulse width in seconds (s)
    - aoa: Angle of Arrival in degrees (0-360)
    - amplitude: Amplitude in dBm
    """
    toa: float
    centre_frequency: float
    pulse_width: float
    aoa: float
    amplitude: float
    emitter_class: int | None = None
