import pandas


def scale_time(dataframe, time_column, time_bucket, aggregation_function_list):
    """Scales timestamp data in a dataframe to a less granular time frequency

    Args:
        dataframe (pandas.Dataframe): pandas dataframe with a timestamp field.
        time_column (string): Name of the time column to scale in the dataframe.
        time_bucket (DateOffset, Timedelta or str): The offset string or object representing target conversion. ex. "2H", "M"
        aggregation_function (List[string]): A list of aggregation functions like sum, average, median, mean, mode, etc. Example: ['sum'], or ['min', 'max', 'sum']
    """

    dataframe[time_column] = pandas.to_datetime(dataframe[time_column])

    # dataframe.set_index(time_column).resample(time_bucket).agg(
    # aggregation_function_list
    # )
    scaled_frame = dataframe.resample(time_bucket, on=time_column).agg(
        aggregation_function_list[0]
    )

    return scaled_frame
