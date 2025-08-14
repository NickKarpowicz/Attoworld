"""Class definition related to atomic unit conversions."""


class AtomicUnits:
    """Defines various physical constants in atomic units.

    Attributes:
      meter: Atomic unit of length in meters.
      nm: Atomic unit of length in nanometers.
      second: Atomic unit of time in seconds.
      fs: Atomic unit of time in femtoseconds.
      Joule: Atomic unit of energy in Joules.
      eV: Atomic unit of energy in electronvolts.
      Volts_per_meter: Atomic unit of electric field in V/m.
      Volts_per_Angstrom: Atomic unit of electric field in V/Angström.
      speed_of_light: Vacuum speed of light in atomic units.
      Coulomb: Atomic unit of electric charge in Coulombs.
      PW_per_cm2_au: PW/cm^2 in atomic units.

    """

    meter: float = 5.2917720859e-11
    nm: float = 5.2917720859e-2
    second: float = 2.418884328e-17
    fs: float = 2.418884328e-2
    Joule: float = 4.359743935e-18
    eV: float = 27.21138383
    Volts_per_meter: float = 5.142206313e11
    Volts_per_Angstrom: float = 51.42206313
    speed_of_light: float = 137.036
    Coulomb: float = 1.60217646e-19
    PW_per_cm2_au: float = 0.1553661415
