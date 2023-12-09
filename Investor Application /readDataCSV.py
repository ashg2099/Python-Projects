"""
Name: Ashwin Gururaj
Student ID: 33921199
Creation Date: 16th Oct 2023
Last Modified Date: 22nd Oct 2023
"""

import pandas as pd
import numpy as np

class readDataCSV:
    def extract_property_info(self, file_path):
        """
        Description: Method to extract property information from a CSV file in the file path.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            DataFrame: The extracted dataframe with property details
        """
        try:
            # Reading the csv file and returning the dataframe
            dataframe = pd.read_csv(file_path)
            print(dataframe)
            return dataframe
        except FileNotFoundError:
            print("File not found. Please provide the correct file path.")
            return None
        except pd.errors.EmptyDataError:
            print("No data is available in the file. Please add the data in the file.")
            return None
        except pd.errors.ParserError:
            print("Unable to parse the data in the file.")
            return None

    def currency_exchange(self, dataframe, exchange_rate):
        """
        Description: To perform currency conversion for property prices in the dataframe

        Args:
            dataframe (pd.DataFrame): The dataframe containing property information extracted from csv data file.
            exchange_rate (float): The exchange rate to apply on each property price.

        Returns:
            np.array: The array of exchanged property prices / tranformed prices
        """
        try:
            # Check if the column named price is there in the datafile or not
            if 'price' not in dataframe.columns:
                raise KeyError("Error encountered: There is no column with the name 'Price'. Kindly check the dataframe.")
            # Else calculated the transformed prices and return the corresponding exchanged rate values
            else:
                exchangedRate = np.array(dataframe['price'].dropna()) * exchange_rate
                return exchangedRate
        except KeyError as k:
            print(f"{k}")
            return None

    def suburb_summary(self, dataframe, suburb):
        """
        Description: To generate a summary(mean,std,min,max,50%) of property information for a specified suburb

        Args:
            dataframe (pd.DataFrame): The dataframe containing property information extracted from csv data file.
            suburb (str): The name of the suburb for which summary needs to be generated

        Returns:
            None
        """
        # Check for null values in the suburb column and drop them
        dataframe.dropna(subset=['suburb'], inplace = True)
        # Checking suburb condition if suburb value is all
        if suburb == "all".lower():
            suburb_summary = dataframe[['bedrooms','bathrooms','parking_spaces']].describe().loc[['mean','std','min','max','50%']]
            print(suburb_summary)
        # Check suburb condition if suburb value is any specific suburb available in the dataframe
        elif suburb.lower() in dataframe['suburb'].str.lower().unique():
            unique_data = dataframe[dataframe['suburb'].str.lower() == suburb.lower()]
            # Calculating the summary based on suburb value.
            suburb_summary = unique_data[['bedrooms','bathrooms','parking_spaces']].describe().loc[['mean','std','min','max','50%']]
            print(suburb_summary)

    def avg_land_size(self, dataframe, suburb):
        """
        Description: To calculate the average land size for a specified suburb or all suburbs

        Args:
            dataframe (pd.DataFrame): The dataframe containing property information.
            suburb (str): The name of the suburb for which to calculate the average land size.

        Returns:
            average land size: The average land size for the specified suburb in sq.mt
        """
        dataframe = dataframe[dataframe['land_size'] >= 0]
        # Checking suburb condition if suburb value is all
        if suburb == "all".lower():
            if (dataframe['land_size_unit'] == 'ha').any():
                dataframe.loc[dataframe['land_size_unit'] == 'ha', 'land_size'] *= 10000
            # Calculating the mean for all the suburbs in the dataframe
            avg_land_size = dataframe['land_size'].mean()
            print("The average land size of all properties is:",avg_land_size,"m²")
            return avg_land_size
        # Check suburb condition if suburb value is any specific suburb available in the dataframe
        elif suburb.lower() in dataframe['suburb'].str.lower().unique():
            if (dataframe['land_size_unit'] == 'ha').any():
                dataframe.loc[dataframe['land_size_unit'] == 'ha', 'land_size'] *= 10000
            # Calculating the mean based on suburb value
            unique_data = dataframe[dataframe['suburb'].str.lower() == suburb.lower()]
            specific_suburb_avg_land_size = unique_data['land_size'].mean()
            print("The average land size of properties in",suburb,":",specific_suburb_avg_land_size,"m²")
            return specific_suburb_avg_land_size

