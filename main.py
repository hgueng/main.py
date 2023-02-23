"""

# import the 7 csv files into pandas dataframes"""
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
df_alpha = pd.read_csv('alpha.csv')
df_bravo = pd.read_csv('bravo.csv')
df_charlie = pd.read_csv('charlie.csv')
df_delta = pd.read_csv('delta.csv')
df_echo = pd.read_csv('echo.csv')
df_foxtrot = pd.read_csv('foxtrot.csv')
df_golf = pd.read_csv('golf.csv')

dataframes = [df_alpha, df_bravo, df_charlie, df_delta, df_echo, df_foxtrot, df_golf]
satellite_names = ['Alpha', 'Bravo', 'Charlie', 'Delta', 'Echo', 'Foxtrot', 'Golf']
telemetry_channels = ['gyro_x','gyro_y','gyro_z','wheel_s','wheel_x','wheel_y','wheel_z','low_voltage','bus_voltage']
Y_title = ['Gyro_x in deg/s','Gyro_y in deg/s','Gyro_z in deg/s','Wheel 1 in rpm','Wheel 2 in rpm','Wheel 3 in rpm','Wheel 4 in rpm','Low Voltage ' , 'Bus Voltage in Volts']



def computeCriticalValues(dataframe=dataframes, subset_name='Insert String' , subset_name2 = 'Insert String'):
    criticalvalues = list()
    gyroratespositive = list()
    gyroratesnegative = list()

    if subset_name in ["wheel_s", "wheel_x", "wheel_y", "wheel_z"]:
        for dataframe in dataframes:
            dataframe1 = dataframe[[subset_name]]
            dataframe1 = dataframe1.dropna(subset=[subset_name])
            x = len(dataframe1) # Available Telemetry points

            dataframe1 = dataframe1.loc[dataframe1[subset_name] >= 8000]
            y = len(dataframe1) # Saturated Wheels
            z = (y*100)/x # Formula to calculate the percentage of points in which condition is true
            criticalvalues.append(z)
    elif subset_name == "low_voltage":
        for dataframe in dataframes:
            dataframe1 = dataframe[[subset_name]]
            dataframe1 = dataframe1.dropna(subset=[subset_name])
            x = len(dataframe1)  # Available Telemetry points
            dataframe1 = dataframe1.loc[dataframe1[subset_name] == 1]
            y = len(dataframe1)  # Saturated Wheels
            z = (y * 100) / x  # Formula to calculate the percentage of points in which condition is true

            criticalvalues.append(z)
    elif subset_name in ["gyro_x", "gyro_y", "gyro_z"]:
        for dataframe in dataframes:
            dataframe1 = dataframe[[subset_name]]
            dataframe1 = dataframe1.dropna(subset=[subset_name])
            x = len(dataframe1)  # Available Telemetry points
            dataframe1 = dataframe1.loc[dataframe1[subset_name] > 2]
            y = len(dataframe1)  # Saturated Wheels
            z = (y * 100) / x  # Formula to calculate the percentage of points in which condition is true
            gyroratespositive.append(z)

            dataframe2 = dataframe[[subset_name]]
            dataframe2 = dataframe2.dropna(subset=[subset_name])
            x2 = len(dataframe2) # Available telemetry points

            dataframe2 = dataframe2.loc[dataframe2[subset_name] < -2]
            y2 = len(dataframe2)    # Gyro Rates over -2 degrees

            z2 = (y * 100) / x2
            gyroratesnegative.append(z2)
    for i in range(len(gyroratespositive)):
        criticalvalues.append(gyroratespositive[i] + gyroratesnegative[i])




    return(criticalvalues)



def showOurCriticalValues():
    wheelsall = computeCriticalValues(dataframes,"wheel_s")
    wheelxall = computeCriticalValues(dataframes,"wheel_x")
    wheelyall = computeCriticalValues(dataframes,"wheel_y")
    wheelzall = computeCriticalValues(dataframes,"wheel_z")
    lowvolt = computeCriticalValues(dataframes,"low_voltage")
    gyrox = computeCriticalValues(dataframes,'gyro_x')
    gyroy = computeCriticalValues(dataframes, 'gyro_y')
    gyroz = computeCriticalValues(dataframes, 'gyro_z')

    #Simple nested loop to compute the average for all wheels combined for each satellite
    #TODO make this better , this looks awful ..maybe insert in function?
    resultwheels = []
    resultgyros = []
    for i in range(len(wheelsall)):
        sum = 0
        for lst in [wheelsall,wheelxall,wheelyall,wheelzall]:
            sum += lst[i]

        resultwheels.append(sum)
    for i in range(len(gyrox)):
        sum = 0
        for lst in [gyrox,gyroy,gyroz]:
            sum += lst[i]
        resultgyros.append(sum)
    resultwheels = [x / 4 for x in resultwheels] # We have 4 Wheels so we need to divide all list elements by 4 to get the arithmetic mean 8)
    resultgyros = [x / 3 for x in resultgyros]
    for i, j in zip(range(0,7), satellite_names):
        print("For satellite " + j + " the average points where the wheels get saturated are: " + format(resultwheels[i] , ".2f") + " %, the average points for gyros over 2 deg. are: "
              + format(resultgyros[i],".2f")," % and "+ format(lowvolt[i],".2f") + " % where the low voltage flag is active." )




def plot_gyro_foreach(dataframes, satellite_names):
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 8))
    for i, df in enumerate(dataframes):
        df = df[['timestamp', 'gyro_x', 'gyro_y', 'gyro_z']]
        df = df.dropna(subset=['gyro_x', 'gyro_y', 'gyro_z'])
        df = df.groupby(df.index // 10000).mean()
        df.insert(0, 'datum', pd.to_datetime(df['timestamp'], unit='ms'))
        df = df[['datum', 'gyro_x', 'gyro_y', 'gyro_z']]
        if not df.empty:
            df.plot(x='datum', y=['gyro_x', 'gyro_y', 'gyro_z'], ax=axs[i//3, i%3])
            axs[i//3, i%3].legend(fontsize=7)
            hourly = mdates.HourLocator(interval=5)
            axs[i//3, i%3].xaxis.set_major_formatter(mdates.DateFormatter('%D-%H:%M'))
            axs[i//3, i%3].set_title('Gyro Telemetry of Satellite ' + satellite_names[i])
            axs[i//3, i%3].set_xlabel('Date')
            axs[i//3, i%3].set_ylabel('Gyro Value')
            if i == 3:  # manually setting y-limits for the affected plot , our problem satellite delta whose gyro is completely out of bounds
                y_ticks = [-4,-3, -2, -1, 0, 1, 2, 3, 4]
                axs[i // 3, i % 3].set_ylim([-4, 4])
                axs[i // 3, i % 3].yaxis.set_ticks(y_ticks)
            else:
                y_ticks = [-0.75,-0.5,-0.25,0,0.25,0.5,0.75] #Issue was, that_yticks was being modified within the loop and the change affects all plots after the third plot
                axs[i // 3, i % 3].yaxis.set_ticks(y_ticks)
        else:
            axs[i//3, i%3].set_visible(False)
    fig.tight_layout()
    for j in range(len(dataframes), 9):
        axs[j//3, j%3].set_visible(False)

    plt.show()
    fig.savefig('GyroValuesForEach.png', dpi=300)


def plot_wheel_foreach(dataframes, satellite_names):
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(16, 10))
    for i, df in enumerate(dataframes):
        df = df[['timestamp', 'wheel_s', 'wheel_x', 'wheel_y', 'wheel_z']]
        df = df.dropna(subset=['timestamp', 'wheel_s', 'wheel_x', 'wheel_y', 'wheel_z'])
        df = df.groupby(df.index // 5000).mean()
        df.insert(0, 'datum', pd.to_datetime(df['timestamp'], unit='ms'))
        df = df[['datum', 'wheel_s', 'wheel_x', 'wheel_y', 'wheel_z']]
        if not df.empty:
            df.plot(x='datum', y=['wheel_s', 'wheel_x', 'wheel_y', 'wheel_z'], ax=axs[i // 3, i % 3])
            if i == 0:
                axs[i // 3, i % 3].legend(fontsize=7)
            else:
                axs[i // 3, i % 3].get_legend().remove()
            hourly = mdates.HourLocator(interval=7)
            axs[i // 3, i % 3].xaxis.set_major_locator(hourly)
            axs[i // 3, i % 3].xaxis.set_major_formatter(mdates.DateFormatter('%D-%H:%M'))
            axs[i // 3, i % 3].set_title('Wheel Telemetry of Satellite ' + satellite_names[i])
            axs[i // 3, i % 3].set_xlabel('Date')
            axs[i // 3, i % 3].set_ylabel('RpM')
        else:
            axs[i // 3, i % 3].set_visible(False)

    fig.tight_layout()
    for j in range(len(dataframes), 9):
        axs[j // 3, j % 3].set_visible(False)
    plt.show()
    fig.savefig('WheelValuesForEach.png' , dpi = 300)

def plot_busvoltage_foreach(dataframes, satellite_names):
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(16, 10))
    for i, df in enumerate(dataframes):
        df = df[['timestamp', 'bus_voltage']]
        df = df.dropna(subset=['timestamp', 'bus_voltage'])
        df = df.groupby(df.index // 5000).mean()
        df.insert(0, 'datum', pd.to_datetime(df['timestamp'], unit='ms'))
        df = df[['datum','bus_voltage']]
        if not df.empty:
            df.plot(x='datum', y='bus_voltage', ax=axs[i // 3, i % 3])
            hourly = mdates.HourLocator(interval=7)
            axs[i // 3, i % 3].xaxis.set_major_locator(hourly)
            axs[i // 3, i % 3].xaxis.set_major_formatter(mdates.DateFormatter('%D-%H:%M'))
            axs[i // 3, i % 3].set_title('Bus Voltage Telemetry of Satellite ' + satellite_names[i])
            axs[i // 3, i % 3].set_xlabel('Date')
            axs[i // 3, i % 3].set_ylabel('Volts')
            axs[i // 3, i % 3].get_legend().remove()
        else:
            axs[i // 3, i % 3].set_visible(False)

    fig.tight_layout()
    for j in range(len(dataframes), 9):
        axs[j // 3, j % 3].set_visible(False)
    plt.show()
    fig.savefig('BusVoltageValuesForEach.png', dpi=300)

def plot_lowvoltage_foreach(dataframes, satellite_names):
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(16, 10))
    for i, df in enumerate(dataframes):
        df = df[['timestamp', 'low_voltage']]
        df = df.dropna(subset=['timestamp', 'low_voltage'])
        df = df.groupby(df.index // 5000).mean()
        df.insert(0, 'datum', pd.to_datetime(df['timestamp'], unit='ms'))
        df = df[['datum','low_voltage']]
        if not df.empty:
            df.plot(x='datum', y='low_voltage', ax=axs[i // 3, i % 3])
            hourly = mdates.HourLocator(interval=7)
            axs[i // 3, i % 3].xaxis.set_major_locator(hourly)
            axs[i // 3, i % 3].xaxis.set_major_formatter(mdates.DateFormatter('%D-%H:%M'))
            axs[i // 3, i % 3].set_title('Low Voltage Telemetry of Satellite ' + satellite_names[i])
            axs[i // 3, i % 3].set_xlabel('Date')
            axs[i // 3, i % 3].set_ylabel('Boolean')
            axs[i // 3, i % 3].get_legend().remove()
        else:
            axs[i // 3, i % 3].set_visible(False)

    fig.tight_layout()
    for j in range(len(dataframes), 9):
        axs[j // 3, j % 3].set_visible(False)
    plt.show()
    fig.savefig('LowVoltageValuesForEach.png', dpi=300)

def plotvaluesforall(dataframes=dataframes, subset_names=telemetry_channels, y_titles=Y_title, sample_size=10000):
    n_plots = len(subset_names)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16, 10))

    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'black']
    labels = ['Alpha', 'Bravo', 'Charlie', "Delta", 'Echo', 'Foxtrot', 'Golf']

    for i, subset_name in enumerate(subset_names):
        row, col = divmod(i, n_cols)
        ax = axs[row, col]

        for index, dataframe in enumerate(dataframes):
            dataframe = dataframe.dropna(subset=[subset_name])
            dataframe = dataframe[['timestamp', subset_name]]
            dataframe = dataframe.groupby(dataframe.index // sample_size).mean()
            dataframe.insert(0, "datum", pd.to_datetime(dataframe['timestamp'], unit='ms'))
            dataframe.plot(x = 'datum', y=subset_name, ax=ax, label=labels[index], color=colors[index])

        ax.set(xlabel='Time', ylabel=y_titles[i], title=f"{y_titles[i]} over time, sample size: {sample_size}")
        ax.legend(fontsize = 5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%D-%H:%M'))
        hourly = mdates.HourLocator(interval=5)
        ax.xaxis.set_major_locator(hourly)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.tick_params(axis='both', which='minor', labelsize=6)

    # remove any unused subplots
    for i in range(n_plots, n_cols * n_rows):
        row, col = divmod(i, n_cols)
        axs[row, col].set_visible(False)

    plt.tight_layout()
    plt.show()
    fig.savefig('AllValuesforAll.png', dpi=300)



def detect_anomaly(subset_name = 'alpha', searchvalue = 'low_voltage' ):
    df = pd.read_csv(f"{subset_name}.csv")
    if searchvalue == 'low_voltage':
        df = df[[searchvalue]]
        df = df.dropna(subset=[searchvalue])
        if df[searchvalue].isin([1]).any().any():
            return True
        else:
            return False
    elif searchvalue == 'bus_voltage':
        df = df[[searchvalue]]
        df = df.dropna(subset=[searchvalue])
        if df[searchvalue].isin([0]).any().any():
            return True
        else:
            return False
    elif searchvalue in ['wheel_s' ,'wheel_x','wheel_y','wheel_z']:
        df = df[[searchvalue]]
        df = df.dropna(subset=[searchvalue])
        zero_percentage = (df[searchvalue] == 0).sum() / len(df) * 100
        if zero_percentage >= 10: #THRESHHOLD TO DETECT IF THE WHEEL ISNT WORKING AT ALL OR JUST STANDING STILL FOR A SHORT PERIOD OF TIME

            return True
        else:
            return False



def plotbetaforAll():
    df = pd.read_csv('beta_sun_deg.csv')
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    fig, ax = plt.subplots(figsize=(9,9))
    df.plot(x='date', y=['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf'], ax=ax)

    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('Beta Sun Degree in Deg')
    # Set xTicks to every 30 days
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    ax.xaxis.set_ticks([df['date'].min()] + ax.get_xticks().tolist())
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.show()
    fig.savefig('BetaAngleForAll.png', dpi=300)
plotbetaforAll()


"""showOurCriticalValues()
plot_busvoltage_foreach(dataframes,satellite_names)
plotvaluesforall(dataframes, telemetry_channels, Y_title, sample_size=12000)
plot_gyro_foreach(dataframes,satellite_names)
plot_wheel_foreach(dataframes, satellite_names)
plot_lowvoltage_foreach(dataframes,satellite_names)"""