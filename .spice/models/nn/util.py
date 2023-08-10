import math
import numpy as np

print('you just imported the util!')
def make_inference_response(predicted: np.ndarray, now: float):
    forecast = []
    for npvalue in predicted:
        # block based model, so time is dense, no "12 sec" timestamp extrapolation required
        now += 1
        value = float(npvalue)
        if math.isnan(value):
            value = -1e10
        forecast.append({
          'timestamp': now,
          'value': value,
        })

    return {
        'forecast': forecast,
    }

