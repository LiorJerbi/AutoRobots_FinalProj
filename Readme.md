# GNSS Raw Measurements Processing

## Overview

This project provides a solution for processing GNSS (Global Navigation Satellite System) raw measurements. It includes functionalities to parse raw measurement log files, compute positioning using a naive algorithm, and generate visualization outputs in KML format. The project has been enhanced with several improvements for better accuracy and usability.

## Requirements

To run this project, you need:

- Python 3.x
- pandas
- numpy
- matplotlib
- navpy
- gnssutils
- simplekml
- georinex
- unlzw3

You can install the dependencies using pip:

```bash
pip install pandas numpy matplotlib navpy gnssutils simplekml georinex unlzw3
```

## Usage
1. Clone this repository:
```bash
git clone https://github.com/LiorJerbi/AutoRobots_Ex1
```
2. Navigate to the project directory:
```bash
cd AutoRobots_Ex1
```

## How to run
```bash
put your gnss raw measurments txt files in the folder `gnss_log_samples`
```

- run cmd in the main folder and write the next code:
```bash
python GnssToPosition.py
```

## Improvements and Add-ons
### 1. Filtering Low-Quality Satellites
Satellites with a Carrier-to-Noise Density Ratio (CN0) below 30 are filtered out to improve the accuracy of the positioning data.

### 2. Ignoring GPS Disruptions
The script identifies and ignores GPS disruptions that incorrectly position the receiver in Beirut or Cairo. This ensures that corrupted data does not affect the final results.

### 3. Weighted Least Squares Algorithm
The GNSS positioning algorithm has been enhanced with a Weighted Least Squares (WLS) approach, which offers significant improvements over the traditional Least Squares (LS) method. Unlike LS, WLS considers the uncertainties (weights) associated with each pseudorange measurement, thereby improving accuracy and robustness in satellite positioning.

#### Key Enhancements:
- **Weighted Optimization:** Incorporates CN0 values as weights to prioritize more reliable satellite signals, enhancing the accuracy of position estimates.
- **Bias Estimation:** Estimates and adjusts biases (`b0`) associated with pseudorange measurements, further refining positioning accuracy.
- **Iterative Refinement:** The output from WLS can be used iteratively to initialize subsequent runs, allowing for continuous improvement in positioning accuracy across multiple iterations.

This approach ensures more accurate and reliable GNSS positioning, suitable for various environmental conditions and satellite signal strengths.

### 4. Live GPS Positioning Using ADB
The project supports sort of live GPS positioning by connecting your phone to your computer using ADB. To enable this feature, follow these steps:

#### Setup Instructions
1. #### Download and Install ADB:
- Download the ADB tools from the [Android Developer website](https://developer.android.com/studio/releases/platform-tools).
- Extract the tools to a known directory (e.g., C:/Users/User/Documents/adb).

2. #### Enable Developer Options and USB Debugging on Your Phone:
- Go to `Settings` > `About Phone` and tap `Build Number` seven times to enable Developer Options.
- Go to `Settings` > `Developer Options` and enable `USB Debugging`.

3. #### Connect Your Phone to Your Computer:
Connect your phone to your computer using a USB cable are ensure both devices are on the same Wi-Fi network to use ADB wirelessly.

4. #### Set Up Wireless ADB:
- Run the following command to enable ADB over TCP/IP:
```bash
adb tcpip 5555
```
- Disconnect the USB cable and connect to your phone's IP address:
```bash
adb connect <your_phone_ip>:5555
```

#### Running the Live Positioning Script
1. #### Find the Latest GNSS Log:
- The script automatically identifies the latest GNSS log file on your phone's /sdcard/Download directory.
2. #### Pull the Latest GNSS Data:
The script pulls the latest GNSS data from your phone to the local gnss_log_samples directory.
3. #### Run the Positioning Algorithm:
Execute the main script and chose option 2 to process the live GNSS data.

## Output
The script generates the following outputs and saves them into `outcomes` folder:
1. `"filename"`.kml: KML file containing the computed path with timestamps for each position.
2. `"filename"`.csv: CSV file with the computed positions and additional columns (Pos.X, Pos.Y, Pos.Z, Lat, Lon, Alt).
- `"filename"` is switched with the original gnss_log file you run your code with.

## Known Issues
- The live positioning feature may experience delays due to network latency when using wireless ADB.
- Some GNSS log files may contain corrupted data that is not yet filtered by the script.

## Contributors
- [Lior Jerbi](https://github.com/LiorJerbi) - 314899493
- [Yael Rosen](https://github.com/yaelrosen77) - 209498211
- [Tomer Klugman](https://github.com/tomerklugman) - 312723612
- [Hadas Evers](https://github.com/hadasevers) - 206398984
