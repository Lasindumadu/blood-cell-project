def detect_leukemia(features):

    abnormal = 0

    for f in features:

        if f["circularity"] < 0.6:
            abnormal += 1

    if abnormal > 10:
        return "Possible Leukemia"

    return "Normal"