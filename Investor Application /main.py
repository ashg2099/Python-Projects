"""
Name: Ashwin Gururaj
Student ID: 33921199
Creation Date: 16th Oct 2023
Last Modified Date: 22nd Oct 2023
"""

from readDataCSV import readDataCSV
from data_visualiser import DataVisualiser

# Function to display main menu of the property investment management system
def mainMenu():
    print("")
    print("╔═════════════════════════════════════════╗")
    print("║  Property Investment Management System  ║")
    print("╠═════════════════════════════════════════╣")
    print("║ 1. Convert currency for property prices ║")
    print("║ 2. Get suburb's property's summary      ║")
    print("║ 3. Discover average land size of suburb ║")
    print("║ 4. Show property value distribution     ║")
    print("║ 5. Track property sales over time       ║")
    print("║ 6. Identify Property by Price           ║")
    print("║ 7. Exit the application                 ║")
    print("╚═════════════════════════════════════════╝")

# Main function where asking for user inputs and process user inputs.
def main():
    readDataDet = readDataCSV()
    dataVisualisation = DataVisualiser()

    # Defining data file path
    dataFilePath = 'property_information.csv'

    # Extracting the data from the data file
    fileData = readDataDet.extract_property_info(dataFilePath)
    while True:

        try:
            mainMenu()

            # Taking the input from user to select the operation that he/she wants to perform
            action_input = int(input("Enter the choice corresponding to your desired option (1-7): "))

            # Processing the inputs given by the user
            if action_input == 1:
                while True:
                    try:
                        # Taking input from user for currency exchange rate
                        currency_exchange_rate = input("Kindly enter the currency exchange rate: ")
                        currency_exchange_rate = float(currency_exchange_rate)

                        # Check if currency exchange rate is negative or not
                        if currency_exchange_rate < 0:
                            raise ValueError
                        if currency_exchange_rate == 0:
                            raise ZeroDivisionError
                        # Else process the input given by user
                        else:
                            converted_prices = readDataDet.currency_exchange(fileData, currency_exchange_rate)
                            print("Conversion is completed. Converted prices are: ","\n",converted_prices)

                        # Exit the loop if operation is successful
                        break
                    except ValueError:
                        print("Error encountered: Exchange rate must be in float or integer. Kindly enter a valid exchange rate.")
                    except ZeroDivisionError:
                        print("Error encountered: The input can't be zero. Please enter a valid exchange rate.")
            elif action_input == 2:
                while True:
                    try:
                        # Taking the input from the user for getting the suburb's summary
                        user_preferred_suburb = input("Enter a suburb to get suburb's properties summary (eg. Clayton, Burwood, Doncaster etc.). If you want to see the summary of properties from all suburbs, kindly type 'all' or 'All': ")

                        # Checking if the suburb value is in alphabets or not and if it is not raising value error
                        if not user_preferred_suburb.isalpha():
                            raise ValueError

                        # Checking if the suburb value is all or any unique suburb or not and if it is not raising value error
                        if user_preferred_suburb.lower()!='all' and user_preferred_suburb.lower() not in fileData['suburb'].str.lower().values:
                            raise ValueError

                        # Processing the user input
                        readDataDet.suburb_summary(fileData, user_preferred_suburb)

                        # Exit the loop if operation is successful
                        break
                    except ValueError as e:
                        print("Error encountered: Invalid suburb name. Kindly enter a valid suburb name (eg. Clayton, Burwood, Doncaster etc.). If you want to see the summary of properties from all suburbs, kindly type 'all' or 'All'.",e)
            elif action_input == 3:
                while True:
                    try:
                        # Taking the input from the user for getting the average land size of the suburb
                        user_preferred_suburb = input("Enter a suburb to get average land size of the properties in the suburb (eg. Clayton, Burwood, Doncaster etc.). If you want to see the avg. land size of properties from all suburbs, kindly type 'all' or 'All': ")

                        # Checking if the suburb value is in alphabets or not and if it is not raising value error
                        if not user_preferred_suburb.isalpha():
                            raise ValueError

                        # Checking if the suburb value is all or any unique suburb or not and if it is not raising value error
                        if user_preferred_suburb.lower()!='all' and user_preferred_suburb.lower() not in fileData['suburb'].str.lower().values:
                            raise ValueError

                        # Processing the user input
                        readDataDet.avg_land_size(fileData, user_preferred_suburb)

                        # Exit the loop if operation is successful
                        break
                    except ValueError as e:
                        print("Error encountered: Invalid suburb name. Kindly enter a valid suburb name (eg. Clayton, Burwood, Doncaster etc.). If you want to see the avg. land size of properties from all suburbs, kindly type 'all' or 'All':",e)
            elif action_input == 4:
                while True:
                    try:
                        # Taking the input from the user for showing the histogram for a suburb
                        user_preferred_suburb = input("Enter a suburb name (eg. Clayton, Burwood, Doncaster etc.). If you want to see the data from all suburbs, kindly type 'all' or 'All': ")

                        # Taking the input from the user for showing the histogram in user preferred currency
                        user_preferred_currency = input("Enter your local currency to see the converted prices in AUD (eg. INR, USD, HKD, SGD, KRW, CNY, GBP, EUR, JPY): ").upper()

                        # Showing the user with the histogram for a specific suburb in user preferred currency
                        dataVisualisation.prop_val_distribution(fileData, user_preferred_suburb, user_preferred_currency)
                        break
                    except ValueError as e:
                        print("Error encountered: ",str(e))
            elif action_input == 5:
                # Displaying the sales trend over the year
                dataVisualisation.sales_trend(fileData)
            elif action_input == 6:
                while True:
                    try:
                        while True:
                        # Taking the input from the user for a suburb name
                            user_preferred_suburb = input("Enter a suburb name (eg. Clayton, Burwood, Doncaster etc.): ").lower()
                            if user_preferred_suburb in fileData['suburb'].str.lower().unique():
                                break
                            else:
                                print(f"The {user_preferred_suburb} is not present in the data. Kindly check the suburb name and enter a valid suburb name.")

                        # Taking the input from the user to get the user preferred budget for property price in a specific suburb
                        user_preferred_budget = input("Please enter your target price of the property: ")
                        user_preferred_budget = int(user_preferred_budget)

                        # Printing the result, True if price is found in the dataframe for a specific suburb and false otherwise
                        result = dataVisualisation.locate_price(user_preferred_budget,fileData,user_preferred_suburb)
                        if not result:
                            print("No properties found for the specified suburb and price range. Please try again.")
                        break
                    except Exception as e:
                        print(f"An unexpected error occurred: {e}")
            elif action_input == 7:
                while True:
                    # Taking input from the input from the user to exit the application or not
                    user_input = input("Are you sure you want to exit the application. Kindly press Y/N: ")

                    # If user selects 'Y' then program will be exited
                    if user_input.lower() == 'y':
                        print("Thanks for using the Investor Management System. Keep investing keep growing !!!")
                        exit()

                    # If user selects 'N' then program will break and return to main menu
                    elif user_input.lower() == 'n':
                        break

                    # If user selects anything else other than 'Y' or 'N' then again ask the user to enter correct input
                    else:
                        print("Invalid option entered. Kindly select either 'Y' or 'N'.")
            else:
                print("Invalid choice selected. Kindly select a number between 1 and 7: ")

        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

# readDataFileContent = readDataCSV()
# dataFileLoc = 'property_information.csv' # Define the file path

# data = readDataFileContent.extract_property_info(dataFileLoc) # Extract Data from the csv file
# Output would be the whole content of the file with 11871 rows and 20 columns

# readDataFileContent.currency_exchange(data,55.0) # transforming the property prices according to given exchange rate
# Output would be the list of prices converted according to the exchange rate -- [53075000. 22275000. 48455000. ... 41690000. 36850000. 28644000.]

# readDataFileContent.suburb_summary(data, 'Clayton') # Get suburb summary and suburb can be changed for eg. Clayton, Burwood or all to see summary of all the suburbs
# Output would be --
#        bedrooms  bathrooms  parking_spaces
# mean   3.127349   1.631886        1.564809
# std    1.585075   1.111867        1.102015
# min    1.000000   1.000000        0.000000
# max   30.000000  20.000000       31.000000
# 50%    3.000000   1.000000        1.000000

#readDataFileContent.avg_land_size(dataframe,'All') # Get suburb avg land size of properties in the suburb and suburb can be changed for eg. Clayton, Burwood or all to see avg land size of properties of all the suburbs
#Output would be -- The average land size of all : 650.4213917367882 m²

# data = readDataFileContent.extract_property_info(dataFileLoc) # Extract Data from the csv file
# dataVisualization = DataVisualiser()
# dataVisualization.prop_val_distribution(data,"Clayton","INR")
# Output would be -- Histogram would be plotted for user preferred currency and user preferred suburb and would be saved as 'property_value_distribution.png'

# dataVisualization.sales_trend(data)
# Output would be -- A line chart for the number of properties sold in each year and would be saved as 'sales_trend.png'

# dataVisualization.locate_price(965000,data,'Burwood')
# Output would be -- True if target price is found particular suburb, False otherwise


