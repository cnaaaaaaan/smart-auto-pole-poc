import subprocess
import time


# Heavy vehicle alarm
def trigger_alarm():

    print("🚨 HEAVY VEHICLE DETECTED!")
    print("🚨 ALARM TRIGGERED!")

    # Play a built-in macOS alarm sound
    subprocess.Popen(
        [
            "afplay",
            "/System/Library/Sounds/Sosumi.aiff"
        ]
    )


# Test alarm
print("Heavy vehicle alarm system started.")
print("Testing alarm...")

trigger_alarm()

time.sleep(3)

print("Alarm test completed.")
