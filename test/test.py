import glob
import sys, os, csv
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import navpy
from gnssutils import ephemeris_manager


LIGHTSPEED = 2.99792458e8

def convertXYZtoLLA(val):
    return navpy.ecef2lla(val)

def ParseToCSV():
    # path = input('pls enter dir path: \n')
    # dirname = os.path.basename(path)
    # filename = os.path.splitext(dirname)[0]
    filename = 'test1'

    fields = ['GPS time', 'SatPRN (ID)', 'Sat.X', 'Sat.Y', 'Sat.Z', 'Pseudo-Range', 'CN0', 'Doppler']

    parent_directory = os.path.split(os.getcwd())[0]
    ephemeris_data_directory = os.path.join(parent_directory, 'data')
    sys.path.insert(0, parent_directory)

    # file_pattern = os.path.join(parent_directory, 'data', 'sample', 'gnss_log_*.txt')
    # matching_files = glob.glob(file_pattern)
    # # Check if any matching files were found
    # if matching_files:
        # If multiple matching files were found, you may choose one or iterate over them
        # input_filepath = matching_files[0]  # Here, we're selecting the first matching file
    #     print("Found matching file:", input_filepath)
    # else:
    #     print("No matching files found.")
    input_filepath = os.path.join(parent_directory, 'data', 'sample', 'gnss_log_2024_04_13_19_51_17.txt')
    # Open the CSV file and iterate over its rows
    with open(input_filepath) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0][0] == '#':
                if 'Fix' in row[0]:
                    android_fixes = [row[1:]]
                elif 'Raw' in row[0]:
                    measurements = [row[1:]]
            else:
                if row[0] == 'Fix':
                    android_fixes.append(row[1:])
                elif row[0] == 'Raw':
                    measurements.append(row[1:])

    android_fixes = pd.DataFrame(android_fixes[1:], columns=android_fixes[0])
    measurements = pd.DataFrame(measurements[1:], columns=measurements[0])

    # Format satellite IDs
    measurements.loc[measurements['Svid'].str.len() == 1, 'Svid'] = '0' + measurements['Svid']
    measurements.loc[measurements['ConstellationType'] == '1', 'Constellation'] = 'G'
    measurements.loc[measurements['ConstellationType'] == '3', 'Constellation'] = 'R'
    measurements['SvName'] = measurements['Constellation'] + measurements['Svid']

    # Remove all non-GPS measurements
    measurements = measurements.loc[measurements['Constellation'] == 'G']

    # Extract SatPRN (ID) from the data
    satPRN = measurements['SvName'].tolist()

    # Convert columns to numeric representation
    measurements['Cn0DbHz'] = pd.to_numeric(measurements['Cn0DbHz'])
    measurements['TimeNanos'] = pd.to_numeric(measurements['TimeNanos'])
    measurements['FullBiasNanos'] = pd.to_numeric(measurements['FullBiasNanos'])
    measurements['ReceivedSvTimeNanos'] = pd.to_numeric(measurements['ReceivedSvTimeNanos'])
    measurements['PseudorangeRateMetersPerSecond'] = pd.to_numeric(measurements['PseudorangeRateMetersPerSecond'])
    measurements['ReceivedSvTimeUncertaintyNanos'] = pd.to_numeric(measurements['ReceivedSvTimeUncertaintyNanos'])

    # A few measurement values are not provided by all phones
    # We'll check for them and initialize them with zeros if missing
    if 'BiasNanos' in measurements.columns:
        measurements['BiasNanos'] = pd.to_numeric(measurements['BiasNanos'])
    else:
        measurements['BiasNanos'] = 0
    if 'TimeOffsetNanos' in measurements.columns:
        measurements['TimeOffsetNanos'] = pd.to_numeric(measurements['TimeOffsetNanos'])
    else:
        measurements['TimeOffsetNanos'] = 0

    measurements['GpsTimeNanos'] = measurements['TimeNanos'] - (
                measurements['FullBiasNanos'] - measurements['BiasNanos'])
    gpsepoch = datetime(1980, 1, 6, 0, 0, 0)
    measurements['UnixTime'] = pd.to_datetime(measurements['GpsTimeNanos'], utc=True, origin=gpsepoch)
    measurements['UnixTime'] = measurements['UnixTime']

    # Split data into measurement epochs
    measurements['Epoch'] = 0
    measurements.loc[
        measurements['UnixTime'] - measurements['UnixTime'].shift() > timedelta(milliseconds=200), 'Epoch'] = 1
    measurements['Epoch'] = measurements['Epoch'].cumsum()

    # Extract GPS time from the data
    gpsTime = measurements['UnixTime'].tolist()

    # Calculate pseudorange in seconds
    WEEKSEC = 604800
    measurements['tRxGnssNanos'] = measurements['TimeNanos'] + measurements['TimeOffsetNanos'] - (measurements['FullBiasNanos'].iloc[0] + measurements['BiasNanos'].iloc[0])
    measurements['GpsWeekNumber'] = np.floor(1e-9 * measurements['tRxGnssNanos'] / WEEKSEC)
    measurements['tRxSeconds'] = 1e-9*measurements['tRxGnssNanos'] - WEEKSEC * measurements['GpsWeekNumber']
    measurements['tTxSeconds'] = 1e-9*(measurements['ReceivedSvTimeNanos'] + measurements['TimeOffsetNanos'])
    measurements['prSeconds'] = measurements['tRxSeconds'] - measurements['tTxSeconds']

    # Convert to meters
    measurements['PrM'] = LIGHTSPEED * measurements['prSeconds']
    measurements['PrSigmaM'] = LIGHTSPEED * 1e-9 * measurements['ReceivedSvTimeUncertaintyNanos']

    # Extract pseudo-range calculations
    pseudo_range = measurements['PrM'].tolist()
    manager = ephemeris_manager.EphemerisManager(ephemeris_data_directory)
    # Calculate satellite Y,X,Z coordinates
    epoch = 0
    num_sats = 0
    while num_sats < 5:
        one_epoch = measurements.loc[
            (measurements['Epoch'] == epoch) & (measurements['prSeconds'] < 0.1)].drop_duplicates(subset='SvName')
        timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
        one_epoch.set_index('SvName', inplace=True)
        num_sats = len(one_epoch.index)
        epoch += 1

    sats = one_epoch.index.unique().tolist()
    ephemeris = manager.get_ephemeris(timestamp, sats)

    def calculate_satellite_position(ephemeris, transmit_time):
        mu = 3.986005e14
        OmegaDot_e = 7.2921151467e-5
        F = -4.442807633e-10
        sv_position = pd.DataFrame()
        sv_position['sv'] = ephemeris.index
        sv_position.set_index('sv', inplace=True)
        sv_position['t_k'] = transmit_time - ephemeris['t_oe']
        A = ephemeris['sqrtA'].pow(2)
        n_0 = np.sqrt(mu / A.pow(3))
        n = n_0 + ephemeris['deltaN']
        M_k = ephemeris['M_0'] + n * sv_position['t_k']
        E_k = M_k
        err = pd.Series(data=[1] * len(sv_position.index))
        i = 0
        while err.abs().min() > 1e-8 and i < 10:
            new_vals = M_k + ephemeris['e'] * np.sin(E_k)
            err = new_vals - E_k
            E_k = new_vals
            i += 1

        sinE_k = np.sin(E_k)
        cosE_k = np.cos(E_k)
        delT_r = F * ephemeris['e'].pow(ephemeris['sqrtA']) * sinE_k
        delT_oc = transmit_time - ephemeris['t_oc']
        sv_position['delT_sv'] = ephemeris['SVclockBias'] + ephemeris['SVclockDrift'] * delT_oc + ephemeris[
            'SVclockDriftRate'] * delT_oc.pow(2)

        v_k = np.arctan2(np.sqrt(1 - ephemeris['e'].pow(2)) * sinE_k, (cosE_k - ephemeris['e']))

        Phi_k = v_k + ephemeris['omega']

        sin2Phi_k = np.sin(2 * Phi_k)
        cos2Phi_k = np.cos(2 * Phi_k)

        du_k = ephemeris['C_us'] * sin2Phi_k + ephemeris['C_uc'] * cos2Phi_k
        dr_k = ephemeris['C_rs'] * sin2Phi_k + ephemeris['C_rc'] * cos2Phi_k
        di_k = ephemeris['C_is'] * sin2Phi_k + ephemeris['C_ic'] * cos2Phi_k

        u_k = Phi_k + du_k

        r_k = A * (1 - ephemeris['e'] * np.cos(E_k)) + dr_k

        i_k = ephemeris['i_0'] + di_k + ephemeris['IDOT'] * sv_position['t_k']

        x_k_prime = r_k * np.cos(u_k)
        y_k_prime = r_k * np.sin(u_k)

        Omega_k = ephemeris['Omega_0'] + (ephemeris['OmegaDot'] - OmegaDot_e) * sv_position['t_k'] - OmegaDot_e * \
                  ephemeris['t_oe']

        sv_position['x_k'] = x_k_prime * np.cos(Omega_k) - y_k_prime * np.cos(i_k) * np.sin(Omega_k)
        sv_position['y_k'] = x_k_prime * np.sin(Omega_k) + y_k_prime * np.cos(i_k) * np.cos(Omega_k)
        sv_position['z_k'] = y_k_prime * np.sin(i_k)
        return sv_position

    sv_position = calculate_satellite_position(ephemeris, one_epoch['tTxSeconds'])

    Yco = sv_position['y_k'].tolist()
    Xco = sv_position['x_k'].tolist()
    Zco = sv_position['z_k'].tolist()


   # Calculate CN0 values
    epoch = 0
    num_sats = 0
    while num_sats < 5:
        one_epoch = measurements.loc[(measurements['Epoch'] == epoch) & (measurements['prSeconds'] < 0.1)].drop_duplicates(subset='SvName')
        timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
        one_epoch.set_index('SvName', inplace=True)
        num_sats = len(one_epoch.index)
        epoch += 1

    CN0 = one_epoch['Cn0DbHz'].tolist()
    doppler = measurements['PseudorangeRateMetersPerSecond'].tolist()




# saving all the above data into csv file
    data = []
    for i in range(len(Yco)):
        row = [gpsTime[i], satPRN[i], Xco[i], Yco[i], Zco[i], pseudo_range[i], CN0[i], doppler[i]]
        data.append(row)

    file_path = os.path.join(parent_directory, filename + '.csv')
    # Write data to CSV file
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header
        writer.writerow(fields)

        # Write the data
        writer.writerows(data)
    return one_epoch['PrM'], sv_position['delT_sv']


prm, delT = ParseToCSV()

parent_directory = os.path.split(os.getcwd())[0]
input_fpath = os.path.join(parent_directory, "test1.csv")

# Initialize lists to store extracted values
sat_x_column = []
sat_y_column = []
sat_z_column = []
pseudorange_column = []
cn0_column = []

# Open the CSV file
with open(input_fpath, newline='') as csvfile:
    # Create a CSV reader object
    reader = csv.reader(csvfile)

    # Read the first row to get the column headers
    headers = next(reader)

    # Find the indices of the required columns
    sat_x_index = headers.index('Sat.X')
    sat_y_index = headers.index('Sat.Y')
    sat_z_index = headers.index('Sat.Z')
    pseudorange_index = headers.index('Pseudo-Range')
    cn0_index = headers.index('CN0')


    # Iterate through each row and extract the values of the required columns
    for row in reader:
        sat_x_column.append(float(row[sat_x_index]))
        sat_y_column.append(float(row[sat_y_index]))
        sat_z_column.append(float(row[sat_z_index]))
        pseudorange_column.append(float(row[pseudorange_index]))
        cn0_column.append(float(row[cn0_index]))

# Convert lists to numpy arrays for easier manipulation
sat_x_array = np.array(sat_x_column)
sat_y_array = np.array(sat_y_column)
sat_z_array = np.array(sat_z_column)
pseudorange_array = np.array(pseudorange_column)
weights = np.array(cn0_column)
# Combine satellite positions into a single variable xs
xs = np.column_stack((sat_x_array, sat_y_array, sat_z_array))

#initial guesses of receiver clock bias and position
b0 = 0
x0 = np.array([0, 0, 0])
pr = prm + LIGHTSPEED * delT
pr = pr.to_numpy()



def weighted_least_squares(xs, measured_pseudorange, x0, b0, weights):
    dx = 100 * np.ones(3)
    b = b0
    G = np.ones((measured_pseudorange.size, 4))
    iterations = 0

    while np.linalg.norm(dx) > 1e-3:
        r = np.linalg.norm(xs - x0, axis=1)
        phat = r + b0
        deltaP = measured_pseudorange - phat

        # Modify residuals using weights
        weighted_residuals = deltaP / np.sqrt(weights)

        # Modify G matrix using weights
        weighted_G = -(xs - x0) / r[:, None] / np.sqrt(weights[:, None])
        G[:, 0:3] = weighted_G

        # Solve weighted least squares
        sol = np.linalg.inv(np.transpose(G) @ G) @ np.transpose(G) @ weighted_residuals

        dx = sol[0:3]
        db = sol[3]
        x0 = x0 + dx
        b0 = b0 + db
    norm_dp = np.linalg.norm(measured_pseudorange - phat)
    return x0, b0, norm_dp

x, b, dp = weighted_least_squares(xs, pr, x0, b0, weights)
print(convertXYZtoLLA(x))
print(b/LIGHTSPEED)
print(dp)
