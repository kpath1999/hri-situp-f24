### Study design

The study would go on for 35 minutes, where no interventions would occur in the first 5 minutes. In the remaining 30 minutes, we would have posture-related interventions if the user is part of the passive or active groups.

Given the time constraints we're facing, we will reduce the number of participants and adjust the factorial design. If the standing desk is not ready as we are close to the deadline, we can 2 users each to the control and passive groups.

- **Control Group:** 2 participants
- **Passive Group:** 4 participants
- **Active Group:** 4 participants

To recap, below is a description of each of the user groups:

1. **Control Group:** They would work on their laptop while we record them with cameras and log all posture-related measurements using Mediapipe.
2. **Passive Group:** About the same set-up as the control group, 15 minutes of vibration cues, and 15 minutes of sound cues. Modality sequencing would be randomized to avoid ordering bias.
3. **Active Group:** Automatic table height adjustments would be made if the user is found to lean too close.

### Experimental Setup

Webcam would be installed on the participant's laptop. Your phone would be set-up using a gooseneck/tripod to view side posture. Another phone would be placed at the bottom of the chair to act as a vibration cue. The script would be run remotely from a slight distance away from the user study to avoid any undue influence.

#### Metrics to Collect

A mixture of objective and subjective metrics would be gathered during the 30-45 minute user session. During this time, our integrated sensor approach will provide an thorough understanding of the system's impact on user posture, comfort, and overall experience.

Below are a few objective metrics that would be considered:

1. `Posture Score:` A composite score will be computed based on data aggregated from the CV system and chair sensors throughout the session. This score will reflect (a) the frequency and duration of poor posture instances, (b) the magnitude of postural deviations, and (c) time spent in optimal posture ranges.
2. `Desk Adjustment/Vibration/Sound Frequency:` The number of autonomous height adjustments made by the system.
3. `Response Time:` How quickly users correct their posture after receiving feedback. A moving average would be used to account for improvements in the score.
TODO: come up with more; also what kind of graphs could we have?


And below are the subjective metrics that would be gathered during the post-study questionnaire:

1. `Perceived Effectiveness:` Users’ perception of how well the system improved their posture.
2. `Comfort:` Physical comfort levels throughout the session.
3. `Ease of Use:` Intuitiveness and non-disruptiveness of the system.
4. `Bodily Fatigue:` Comparison of fatigue levels before and after the session.
5. `Perceived Productivity:` Users' perception of how the system affected their work efficiency.
6. `Likeability:` Overall satisfaction with the system.
7. `Perceived Intelligence:` Users’ view on how smart or adaptive they found the system.
8. `Intrusiveness:` Degree to which users felt the system interrupted their workflow.