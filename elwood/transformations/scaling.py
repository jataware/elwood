import pandas


def scale_time(
    dataframe, time_column, time_bucket, aggregation_functions, geo_columns=None
):
    """Scales timestamp data in a dataframe to a less granular time frequency

    Args:
        dataframe (pandas.Dataframe): pandas dataframe with a timestamp field.
        time_column (string): Name of the time column to scale in the dataframe.
        time_bucket (DateOffset, Timedelta or str): The offset string or object representing target conversion. ex. "2H", "M"
        aggregation_functions (List[string] or Dict{str:str}): A list containing one aggregation function, or a Dict containing column
            names and aggregation functions like sum, average, median, mean, mode, etc. Example: ['sum'], or {'temp': 'min', 'rainfall': 'max', 'visitors': 'sum'}
        geo_columns (List[string]): A list of the geographical columns that should not be modified by the transformation.
    """

    dataframe[time_column] = pandas.to_datetime(dataframe[time_column])

    if geo_columns:
        dataframe = dataframe.set_index(geo_columns)

        dataframe = dataframe.groupby(geo_columns)

    aggregator = aggregation_functions
    if isinstance(aggregation_functions, list):
        aggregator = aggregation_functions[0]
    scaled_frame = dataframe.resample(time_bucket, on=time_column).agg(aggregator)

    # scaled_frame.columns = scaled_frame.columns.get_level_values(0)
    scaled_frame.reset_index(inplace=True)

    return scaled_frame
