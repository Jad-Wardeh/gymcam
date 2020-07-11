# gymcam
## Description
In this project, my overall goal was to assist individuals in the gym by employing Raspberry Pi and Camera technology into practical use. More specifically, this design is intended to aid one in improving how they do squats by allowing them to view their form from a different angle. Simultaneously, with using OpenPose, one is able to view lines on their screen according to their body, forming a precise image of what they look like while working out. At the gym, the camera will be placed to the side, and the video will be streamed directly to their phone, which is placed in front of the individual.
## What Was Used
-  Raspberry Pi 3 B+ - comes along with 1.4GHz 64-bit quad-core ARM Cortex-a53 CPU, ethernet, Availability for LAN and Bluetooth connections, and various other features.
https://github.com/Jad-Wardeh/gymcam/blob/master/img/RasPi3.png
-  Pi Camera 2 - allowed me to stream videos with pretty good quality, containing a 8-megapixel sensor, and allowed me to run OpenCV, OpenPose, and TriangleSolver on it.
https://github.com/Jad-Wardeh/gymcam/blob/master/img/RasCameraV2.png
-  Intel Neural Compute Stick 2 - an NPU that allowed the camera to become much more responsive and much smoother. The Raspberry Pi did not generate enough power to allow the camera to run smoothly.

