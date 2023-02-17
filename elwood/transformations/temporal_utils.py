def calculate_temporal_boundary(dataframe, time_column):
    return {"min": min(dataframe[time_column]), "max": max(dataframe[time_column])}
