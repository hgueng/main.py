
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import time



start_time = time.time()
# import the 7 csv files into pandas dataframes
df_alpha = pd.read_csv('alpha.csv')
df_bravo = pd.read_csv('bravo.csv')
df_charlie = pd.read_csv('charlie.csv')
df_delta = pd.read_csv('delta.csv')
df_echo = pd.read_csv('echo.csv')
df_foxtrot = pd.read_csv('foxtrot.csv')
df_golf = pd.read_csv('golf.csv')


# These lists are used in the functions later on.
dataframes = [df_alpha, df_bravo, df_charlie, df_delta, df_echo, df_foxtrot, df_golf]
satellite_names = ['Alpha', 'Bravo', 'Charlie', 'Delta', 'Echo', 'Foxtrot', 'Golf']
telemetry_channels = ['gyro_x','gyro_y','gyro_z','wheel_s','wheel_x','wheel_y','wheel_z','low_voltage','bus_voltage']
Y_title = ['Gyro_x in deg/s','Gyro_y in deg/s','Gyro_z in deg/s','Wheel 1 in rpm','Wheel 2 in rpm'
          ,'Wheel 3 in rpm','Wheel 4 in rpm','Low Voltage ' , 'Bus Voltage in Volts']



def computeCriticalValues(dataframe, subset_name='Insert String' , subset_name2=''):
    criticalvalues = list()
    gyroratespositive = list()
    gyroratesnegative = list()
    # This function takes a string / in this case telemtry channel as input
    # and then computes the critical values for this telemtry channel for each satellite
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
    # Here we have to be careful, I think it is required to include -2 degs aswell.
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
    # with our returned list of Critical Values this function, combines all the crit. values
    # for wheels and gyros together in an average and also prints out our wanted percentage
    wheelsall = computeCriticalValues(dataframes,"wheel_s")
    wheelxall = computeCriticalValues(dataframes,"wheel_x")
    wheelyall = computeCriticalValues(dataframes,"wheel_y")
    wheelzall = computeCriticalValues(dataframes,"wheel_z")
    lowvolt = computeCriticalValues(dataframes,"low_voltage")
    gyrox = computeCriticalValues(dataframes,'gyro_x')
    gyroy = computeCriticalValues(dataframes, 'gyro_y')
    gyroz = computeCriticalValues(dataframes, 'gyro_z')

    #Simple nested loop to compute the average for all wheels + gyros combined for each satellite
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
    # Our function for plotting the 3 Gyros for each satellite in a single page
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 8))
    for i, df in enumerate(dataframes):
        # Selecting our wanted cols and also dropping NaN values.
        df = df[['timestamp', 'gyro_x', 'gyro_y', 'gyro_z']]
        df = df.dropna(subset=['gyro_x', 'gyro_y', 'gyro_z'])
        # grouping data and calculating mean of these groups.
        # reducing the points to plot to  about ~ 70 points which makes the
        # graph smooth and presentable. Same method is used in most of the other plots too, except for beta dun deg.
        df = df.groupby(df.index // 10000).mean()
        df.insert(0, 'datum', pd.to_datetime(df['timestamp'], unit='ms'))
        df = df[['datum', 'gyro_x', 'gyro_y', 'gyro_z']]
        if not df.empty:
            df.plot(x='datum', y=['gyro_x', 'gyro_y', 'gyro_z'], ax=axs[i//3, i%3])
            axs[i//3, i%3].legend(fontsize=7)
            hourly = mdates.HourLocator(interval=6) #X-Ticks set to every 6 hours
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
            # removing unused plots
            axs[i//3, i%3].set_visible(False)
    fig.tight_layout()
    for j in range(len(dataframes), 9):
        axs[j//3, j%3].set_visible(False)

    #plt.show()
    fig.savefig('GyroValuesForEach.png', dpi=300)


def plot_wheel_foreach(dataframes, satellite_names):
    # Same procedure as above
    # I divided all wanted telemetry channels into different functions to reduce compute time
    # Also you can just simply call the function for your wanted tel channel.
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
            hourly = mdates.HourLocator(interval=6)
            #The expression i // 3 is used to calculate the row index of the current subplot, while i % 3 is used to calculate the column index of the current subplot.
            # This is necessary because the subplots are arranged in a grid with 3 rows and 3 columns, and the i index is used to iterate over the different satellite datasets.
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
    #plt.show()
    fig.savefig('WheelValuesForEach.png' , dpi = 300)

def plot_busvoltage_foreach(dataframes, satellite_names):
    # Same as above
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(16, 10))
    for i, df in enumerate(dataframes):
        df = df[['timestamp', 'bus_voltage']]
        df = df.dropna(subset=['timestamp', 'bus_voltage'])
        df = df.groupby(df.index // 5000).mean()
        df.insert(0, 'datum', pd.to_datetime(df['timestamp'], unit='ms'))
        df = df[['datum','bus_voltage']]
        if not df.empty:
            df.plot(x='datum', y='bus_voltage', ax=axs[i // 3, i % 3])
            hourly = mdates.HourLocator(interval=6)
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
    #plt.show()
    fig.savefig('BusVoltageValuesForEach.png', dpi=300)

def plot_lowvoltage_foreach(dataframes, satellite_names):
    # Same as above
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(16, 10))
    for i, df in enumerate(dataframes):
        df = df[['timestamp', 'low_voltage']]
        df = df.dropna(subset=['timestamp', 'low_voltage'])
        df = df.groupby(df.index // 5000).mean()
        df.insert(0, 'datum', pd.to_datetime(df['timestamp'], unit='ms'))
        df = df[['datum','low_voltage']]
        if not df.empty:
            df.plot(x='datum', y='low_voltage', ax=axs[i // 3, i % 3])
            hourly = mdates.HourLocator(interval=6)
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
    #plt.show()
    fig.savefig('LowVoltageValuesForEach.png', dpi=300)

def plotvaluesforall(dataframes=dataframes, subset_names=telemetry_channels, y_titles=Y_title, sample_size=10000):

    # Calculating how many cols and rows are needed, can differ because of more telem. channels
    n_plots = len(subset_names)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16, 10))
    #Edit this or put it inside params of function if there are more satellites to analyze in the future
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'black']
    labels = ['Alpha', 'Bravo', 'Charlie', "Delta", 'Echo', 'Foxtrot', 'Golf']

    for i, subset_name in enumerate(subset_names):
        row, col = divmod(i, n_cols) #The divmod function is used to calculate the row and column indices for the current plot
        # based on the index i and the number of columns n_cols. The divmod function returns a tuple of two values: the quotient and the remainder
        # of the division of the two inputs. In this case,
        # divmod(i, n_cols) calculates the row and column indices for the plot, and those values are assigned to the variables row and col.
        ax = axs[row, col]

        for index, dataframe in enumerate(dataframes):
            dataframe = dataframe.dropna(subset=[subset_name])
            dataframe = dataframe[['timestamp', subset_name]]
            dataframe = dataframe.groupby(dataframe.index // sample_size).mean()
            # Inserting a converted date column from the unix timestamps
            dataframe.insert(0, "datum", pd.to_datetime(dataframe['timestamp'], unit='ms'))
            dataframe.plot(x = 'datum', y=subset_name, ax=ax, label=labels[index], color=colors[index])

        ax.set(xlabel='Time', ylabel=y_titles[i], title=f"{y_titles[i]} over time, sample size: {sample_size}")
        ax.legend(fontsize = 5)
        # Customizing xticks and intervals. I chose every 6 hours.
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%D-%H:%M'))
        hourly = mdates.HourLocator(interval=6)
        ax.xaxis.set_major_locator(hourly)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.tick_params(axis='both', which='minor', labelsize=6)

    # Removing any unused subplots at the end
    for i in range(n_plots, n_cols * n_rows):
        row, col = divmod(i, n_cols)
        axs[row, col].set_visible(False)

    plt.tight_layout()
    #plt.show()
    fig.savefig('AllValuesforAll.png', dpi=300)



def detect_anomaly(subset_name = 'alpha', searchvalue = 'low_voltage' ):
    df = pd.read_csv(f"{subset_name}.csv")

    # Searching if there's a low voltage alarm in the measured time.
    if searchvalue == 'low_voltage':
        df = df[[searchvalue]]
        df = df.dropna(subset=[searchvalue])
        if df[searchvalue].isin([1]).any().any():
            return True
        else:
            return False
    # With bus voltage as search parameter we can see if the voltage is or has
    # dropped to 0
    elif searchvalue == 'bus_voltage':
        df = df[[searchvalue]]
        df = df.dropna(subset=[searchvalue])
        if df[searchvalue].isin([0]).any().any():
            return True
        else:
            return False
    # Testing if wheel has failed or just standing still
    # I tested out different threshholds and 10 is optimal in my opinion for this kind
    # of anomaly detection
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
    # Custom Linestyle and markes for each graph
    markers = ['o', '^', 's', 'd', 'v', 'd', 'p']
    linestyles = ['-', '--', '-.', ':', '-', '-.', '--']
    markevery_values = [[50,150, 250, 350], [60,120, 240, 360], [50,100, 180, 260], [70,130, 260, 320], [40,90, 170, 250],
                        [50,120, 220, 320], [80, 140, 200,310]]
    colors = ['red', 'blue', 'green', 'gray', 'orange', 'purple', 'pink', 'black']
    # Making satellite graphs with the same beta sun degree
    # visible with markers so we can identify which satellite is in which flock
    fig, ax = plt.subplots(figsize=(12, 9))

    for i, col in enumerate(['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf']):
        df.plot(x='date', y=col, ax=ax, marker=markers[i], linestyle=linestyles[i], label=col , markevery = markevery_values[i] , color = colors[i],
                markersize = 7)

    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('Beta Sun Degree in Deg')
    # Setting xTicks to every 30 days here
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    ax.xaxis.set_ticks([df['date'].min()] + ax.get_xticks().tolist())
    ax.set_title('Beta Sun Degree over Time')
    # Rotating x axis dates
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.grid('on')
    #plt.show()
    fig.savefig('BetaAngleForAll.png', dpi=300)

# Note: I commented out plt.show() on functions to measure runtime like asked.
#================================== First Question =========================================================================
# All the telemetry channels of each satellite in a single page (all gyros and wheels together)
plot_gyro_foreach(dataframes,satellite_names)
plot_wheel_foreach(dataframes, satellite_names)
plot_lowvoltage_foreach(dataframes,satellite_names)
plot_busvoltage_foreach(dataframes,satellite_names)
first_end_time = time.time()
first_total_time = first_end_time - start_time
print(f"Runtime: for Question 1 a): {first_total_time:,.2f} seconds")
#All the telemetry channels of all the satellites combined in a single page
plotvaluesforall(dataframes, telemetry_channels, Y_title, sample_size=12000)
second_end_time = time.time()
second_total_time = second_end_time - first_end_time
print(f"Runtime: for Question 1 b): {second_total_time:,.2f} seconds")
#The beta angle of all the satellites, combined in a single graph
plotbetaforAll()
third_end_time = time.time()
third_total_time = third_end_time - second_end_time
print(f"Runtime: for Question 1 c): {third_total_time:,.2f} seconds")
#================================== Second Question =========================================================================
showOurCriticalValues()
fourth_end_time = time.time()
fourth_total_time = fourth_end_time - third_end_time
print(f"Runtime: for Question 2): {fourth_total_time:,.2f} seconds")
end_time = time.time()
total_time = end_time - start_time
print(f"Total runtime: {total_time:,.2f} seconds")