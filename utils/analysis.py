def analyze_prediction(label, prob):
    # Adjust confidence based on class
    confidence = prob if label == "Pneumonia" else 1 - prob
    confidence_pct = round(confidence * 100, 2)

    # Status (uncertainty handling)
    if confidence > 0.75:
        status = "Reliable"
    elif confidence > 0.4:
        status = "Uncertain"
    else:
        status = "Low Confidence"

    # Risk level
    if label == "Pneumonia":
        if confidence > 0.8:
            risk = "High"
        elif confidence > 0.5:
            risk = "Moderate"
        else:
            risk = "Low"
    else:
        risk = "Low"

    return confidence_pct, risk, status