from pydantic import BaseModel, Enum, Field
from typing import Optional, List, Dict, Any


class ColumnType(str, Enum):
    DATE = "date"
    GEO = "geo"
    FEATURE = "feature"


class FeatureType(str, Enum):
    INT = "int"
    FLOAT = "float"
    STR = "str"
    BINARY = "binary"
    BOOLEAN = "boolean"


class GeoType(str, Enum):
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    COORDINATES = "coordinates"
    COUNTRY = "country"
    ISO2 = "iso2"
    ISO3 = "iso3"
    STATE = "state/territory"
    COUNTY = "county/district"
    CITY = "municipality/town"


class DateType(str, Enum):
    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    EPOCH = "epoch"
    DATE = "date"


class PrimaryGeography(BaseModel):
    name: str
    display_name: str
    description: str
    geo_type: GeoType
    is_geo_pair: str


class PrimaryDatetime(BaseModel):
    name: str
    display_name: str
    description: str
    date_type: DateType
    time_format: str = Field(
        title="Time Format",
        description="The strftime formatter for this field",
        example="%y",
    )


class Features(BaseModel):
    name: str
    display_name: str
    description: str
    type: ColumnType
    feature_type: FeatureType
    units: str
    units_description: str


class Qualifiers(BaseModel):
    name: str
    display_name: str
    description: str
    type: ColumnType
    feature_type: FeatureType
    units: str
    units_description: str
    qualifies: list[str]


class DatasetAnnotation(BaseModel):
    date: List[PrimaryDatetime]
    geo: List[PrimaryGeography]
    features: List[Features]
    qualifiers: Optional[List[Qualifiers]]
