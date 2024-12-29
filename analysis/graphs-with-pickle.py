# graph.py graph outputs from a data recording

import numpy
import matplotlib.pyplot as plt
import pickle as pkl

# TODO: Change path to new pickle structure 
filename = "test_posture_data_2024-11-05-10-23-41.pkl"


with open(f'logs/{filename}', 'rb') as file:
    data = pkl.load(file)


plt.figure(figsize=(12, 4))  # Adjusted figure size for better visibility

plt.subplot(241)
plt.title("Neck Inclination")
plt.plot(data['measured_neck_inclination'], color='blue')
plt.xlabel('Time')
plt.ylabel('Inclination (degrees)')

plt.subplot(242)
plt.title("Torso Inclination")
plt.plot(data['measured_torso_inclination'], color='orange')
plt.xlabel('Time')
plt.ylabel('Inclination (degrees)')

plt.subplot(243)
plt.title("Level")
plt.plot(data['measured_level'], color='green')
plt.xlabel('Time')
plt.ylabel('Level (m)')

plt.subplot(244)
plt.title("Lean")
plt.plot(data['measured_lean'], color='red')
plt.xlabel('Time')
plt.ylabel('Lean (m)')

plt.subplot(245)
plt.title("Pred Score")
plt.plot(data['measured_eval_score'], color='purple')
plt.xlabel('Time')
plt.ylabel('Score')

plt.subplot(246)
plt.title("True Score")
plt.plot(data['measured_true_score'], color='brown')
plt.xlabel('Time')
plt.ylabel('Score')

plt.subplot(247)
plt.title("Spine Length")
plt.plot(data['measured_spine_length'], color='pink')
plt.xlabel('Time')
plt.ylabel('Length (cm)')

plt.subplot(248)
plt.title("Vibration")
plt.plot(data['sent_vibration'], color='red')
plt.xlabel('Time')
plt.ylabel('Vibrated?')

plt.subplot(249)
plt.title("Sound")
plt.plot(data['sent_sound'], color='red')
plt.xlabel('Time')
plt.ylabel('Dinged?')

# Adjusting layout for better spacing
plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Increase vertical and horizontal spacing
plt.tight_layout()
plt.show()