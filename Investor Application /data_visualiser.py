"""
Name: Ashwin Gururaj
Student ID: 33921199
Creation Date: 16th Oct 2023
Last Modified Date: 22nd Oct 2023
"""

import matplotlib.pyplot as plt
import pandas as pd
from readDataCSV import readDataCSV

class DataVisualiser:

    def prop_val_distribution(self, dataframe, suburb, target_currency = "AUD"):
        """
        Description: To generate a histogram of property values for a specified suburb or for all suburbs

        Args:
            dataframe (pd.DataFrame): The dataframe containing property information.
            suburb (str): The name of the suburb for which the histogram needs to be generated
            target_currency (str): The target currency for conversion (default is "AUD").

        Returns:
            None
        """

        readDataFile = readDataCSV()
        # Dictionary for currency exchange rates which is a local variable
        currency_dict = {"AUD": 1, "USD": 0.66, "INR": 54.25, "CNY": 4.72,
                        "JPY": 93.87, "HKD": 5.12, "KRW": 860.92, "GBP": 0.51,
                        "EUR": 0.60, "SGD": 0.88}

        # Check if target currency is in the dictionary or not
        if target_currency not in currency_dict:
            print(f"No valid or matching currency with the name '{target_currency}' found. Displaying the histogram in 'AUD'.")
            target_currency = 'AUD'

        # Check if column named price is there in the dataframe or not
        if 'price' not in dataframe.columns:
            print("Unable to locate a column named 'price' in the dataframe. Not able to generate histogram.")

        # Check if all the values in the datafframe are empty or not
        if dataframe['price'].isnull().all():
            print("No values found for price in the dataframe. Not able to generate histogram.")

        # Check if suburb value is all and set displayed data based on the suburb value
        if suburb == 'all'.lower():
            print("Showing results for all the suburbs.")
            displayedData = dataframe

        # Check if suburb value is any specific suburb and set displayed data based on the suburb value
        elif suburb.lower() in dataframe['suburb'].str.lower().unique():
            print(f"Showing results for {suburb}.")
            displayedData = dataframe[dataframe['suburb'].str.lower() == suburb.lower()]

        # Check if suburb value is invalid and if it is then displayeddata is set to dataframe and shown result for all suburb value
        else:
            print(f"{suburb} is not available in the dataframe. Showing the result for all the suburbs.")
            displayedData = dataframe

        # Converting the property prices based on the target currency's exchange rate value
        converted_property_prices = readDataFile.currency_exchange(displayedData, currency_dict[target_currency])
        converted_property_prices = pd.DataFrame(converted_property_prices).reset_index()
        converted_property_prices = converted_property_prices.rename(columns={0:'price'})

        # Plotting the histogram
        plt.hist(converted_property_prices['price'], edgecolor = 'black')
        plt.xlabel("Property Value in "+target_currency.upper())
        plt.ylabel("Total number of Properties")
        plt.title("PROPERTY VALUE DISTRIBUTION FOR "+suburb.upper()+" SUBURB")
        plt.savefig("property_value_distribution.png")

        # Uncomment below line of code to display the histogram
        # plt.show()

    def sales_trend(self,dataframe):
        """
        Description: To visualize the sales trend over the years using a line chart.

        Args:
            dataframe (pd.DataFrame): The dataframe containing property information.

        Returns:
            None
        """
        try:
            # Check if the 'sold_date' column exists in the dataframe
            if 'sold_date' not in dataframe.columns:
                raise ValueError("The 'sold_date' column is missing in the dataframe.")

            # Converting sold_date to datetime and handling errors if any using coerce
            dataframe['sold_date'] = pd.to_datetime(dataframe['sold_date'], errors='coerce')

            # Calculate the number of properties sold in each year
            sales_per_year = dataframe['sold_date'].dt.year.value_counts().sort_index()

            # Check if there is data available for the sales trend each year
            if sales_per_year.empty:
                raise ValueError("No sales data available to generate the sales trend.")

            # Visualize the sales trend as a line chart
            plt.plot(sales_per_year.index, sales_per_year.values, marker = 'o')
            plt.xlabel("Year")
            plt.ylabel("Number of Properties Sold")
            plt.title("Sales Trend")
            plt.grid()
            plt.savefig("sales_trend.png")

            # Uncomment below line of code to display the line chart
            #plt.show()

        # Exception handling for the exceptions happening if any
        except ValueError as ve:
            print("Error:", ve)

        except Exception as e:
            print("An error occurred:", e)

    def locate_price(self, target_price, dataframe, target_suburb):
        """
        Description: To find out if a specific value is in the list of prices for a specific suburb.

        Args:
            target_price (int): The target price to search for.
            dataframe (pd.DataFrame): The dataframe containing property information from the csv file.
            target_suburb (str): The target suburb to filter the data.

        Returns:
            bool: True if the target price is found, False otherwise.
        """
        try:
            # Drop if any null values in the price column
            dataframe.dropna(subset=['price'], inplace=True)
            # Get the list of prices for the properties in the target suburb
            filtered_data = dataframe[dataframe['suburb'].str.lower() == target_suburb.lower()]
            prices = list(map(int, filtered_data['price'].tolist()))
            # Check if the list of prices is not empty
            if not prices:
                print(f"No prices found for suburb '{target_suburb}'.")
                return False

            # Sort the prices of the property in the dataframe in descending order using reverse insertion sort
            self.reverse_insertion_sort(prices)

            # Using the binary search to find the target price and return boolean based on that
            result = self.binary_search(prices, target_price)
            print(result)
            return result

        except KeyError:
            print("The 'price' column is not found in the dataframe.")
            return False
        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    def reverse_insertion_sort(self, array):
        """
        Description: To perform reverse insertion sort on the given array.

        Args:
            array (list): The list to be sorted in descending order.

        Returns:
            None
        """
        for m in range(1, len(array)):
            key = array[m]
            n = m - 1
            while n >= 0 and key > array[n]:
                array[n + 1] = array[n]
                n -= 1
            array[n + 1] = key

    def binary_search(self, arr, target):
        """
        Description: To perform recursive binary search on the given array.

        Args:
            arr (list): The sorted list to search.
            target (int): The target value to search for.

        Returns:
            bool: True if the target value is found and if not found return False.
        """
        if not arr:
            return False
        else:
            mid_val = len(arr) // 2
            if arr[mid_val] == target:
                return True
            elif arr[mid_val] > target:
                return self.binary_search(arr[mid_val + 1:], target)
            else:
                return self.binary_search(arr[:mid_val], target)

