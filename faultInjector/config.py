


class ConfigInjector:
    startFault = 100
    numFaults = 1 # 1000
    duration = 1
    nameLayer = ["relu3"]
    # nameLayer = ["relu3", "relu2", "relu1"]
    # "impulsFunction", "randomFunction", "zeroFunction", "valueFunction", "magnitudeFunction"
    function = "impulsFunction"
    faultValue = 10 # 10000