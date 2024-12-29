import os 
import sys
import pandas
import numpy as np
from scipy.stats import mannwhitneyu, kruskal, tukey_hsd

import matplotlib.pyplot as plt

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

baseline_list = []
vibration_list = []
sound_list = []

before_noise = []
after_noise = []

before_vibration = []
after_vibration = []

for filename in os.listdir(os.path.join(root, "logs", "analysis", "passive_group")):
    if not filename.endswith(".csv"):
        continue
    data = pandas.read_csv(os.path.join(root, "logs", "analysis", "passive_group", filename))

    # first we need to get where we start sending values
    sent_noise = np.where(data.sent_sound == 1)[0]
    sent_vibration = np.where(data.sent_vibration == 1)[0]
    first_noise = sent_noise[0].item() - 60*2
    first_vibration = sent_vibration[0].item() - 60*2

    # average composite score during no intervention
    baseline = np.mean(data.measured_composite_score[:first_noise]).item()  # start to a minute before the first noise 
    baseline_list.append(baseline)

    # average composite score during each intervention method
    if first_noise < first_vibration:
        m1 = np.mean(data.measured_composite_score[first_noise:first_vibration]).item()
        m2 = np.mean(data.measured_composite_score[first_vibration:]).item()
        sound_list.append(m1)
        vibration_list.append(m2)
    else:
        m1 = np.mean(data.measured_composite_score[first_vibration:first_noise]).item()
        m2 = np.mean(data.measured_composite_score[first_noise:]).item()
        vibration_list.append(m1)
        sound_list.append(m2)

    # compare before and after intervention
    noise_a = []
    noise_b = []
    vibration_a = []
    vibration_b = []
    time_range = 50
    for i in sent_noise:
        noise_b.append(np.mean(data.measured_composite_score[i-time_range:i]).item())
        noise_a.append(np.mean(data.measured_composite_score[i:i+time_range]).item())
    before_noise.append(np.mean(noise_b).item())
    after_noise.append(np.mean(noise_a).item())

    for i in sent_vibration:
        vibration_b.append(np.mean(data.measured_composite_score[i-time_range:i]).item())
        vibration_a.append(np.mean(data.measured_composite_score[i:i+time_range]).item())
    before_vibration.append(np.mean(vibration_b).item())
    after_vibration.append(np.mean(vibration_a).item())

if True:
    print("Baseline score:", baseline_list)
    print("Sound Intervention score:", sound_list)
    print("Vibratioin Intervention score:", vibration_list)
    print("Score before noise:", before_noise)
    print("Score after noise:", after_noise)
    print("Score before vibration:", before_vibration)
    print("Score after vibration:", after_vibration)

# stats test
print("Control/sound/vibration p-value:", kruskal(baseline_list, sound_list, vibration_list).pvalue)
print("Before/after noise p-value:", mannwhitneyu(before_noise, after_noise).pvalue)
print("Before/after vibration p-value:", mannwhitneyu(before_vibration, after_vibration).pvalue)

# tukey
results = tukey_hsd(baseline_list, sound_list, vibration_list)
print(results)

fig, ax = plt.subplots(1, 1)
ax.boxplot([baseline_list, sound_list, vibration_list])
ax.set_xticklabels(["baseline", "Sound", "Vibration"])
ax.set_ylabel("Means")
plt.show()