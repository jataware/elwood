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

    matching_return = {"exact_match": [], "fuzzy_match": []}
    for entry in non_matched:
        try:
            matches = fuzzyprocess.extract(entry, gadm_list, scorer=fuzz.partial_ratio)
            formatted_response, exact_match = construct_gadm_response(matches)
            formatted_response["raw_value"] = entry
            if exact_match:
                matching_return["exact_match"].append(formatted_response)
            else:
                matching_return["fuzzy_match"].append(formatted_response)
        except Exception as e:
            matches = None
            logging.error(f"Error in match_geo_names: {e}")

    return matching_return


def construct_gadm_response(matches_list):
    exact_match = False

    first_match = matches_list[0]
    if first_match[1] == 100:
        exact_match = True
        matches_response = {
            "raw_value": "",
            "gadm_resolved": first_match[0],
            "confidence": first_match[1],
            "alternatives": [name for name, value in matches_list],
        }
        return matches_response, exact_match
    matches_response = {
        "raw_value": "",
        "gadm_resolved": first_match[0],
        "confidence": first_match[1],
        "alternatives": [name for name, value in matches_list],
    }
    return matches_response, exact_match
