import logging
import pandas

from fuzzywuzzy import fuzz
from fuzzywuzzy import process as fuzzyprocess
import geofeather as gf


def gadm_fuzzy_match(dataset, geo_column, geofeather_object, admin_level):
    gadm_list = geofeather_object[admin_level].unique()
    print(gadm_list)

    non_matched = dataset[geo_column].unique()

    print(non_matched)

    matching_return = {}
    for entry in non_matched:
        try:
            matches = fuzzyprocess.extract(entry, gadm_list, scorer=fuzz.partial_ratio)
        except Exception as e:
            matches = None
            logging.error(f"Error in match_geo_names: {e}")

        matching_return[entry] = matches

    return matching_return
