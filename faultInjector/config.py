


class ConfigInjector:
    startFault = 30
    numFaults = 1000
    duration = 20
    nameLayer = ["relu3"]
    # "impulsFunction", "randomFunction", "zeroFunction", "valueFunction", "magnitudeFunction"
    function = "impulsFunction"
    faultValue = 1000