from enum import Enum
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


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


class FileType(str, Enum):
    CSV = "csv"
    GEOTIFF = "geotiff"
    EXCEL = "excel"
    NETCDF = "netcdf"


class PrimaryGeography(BaseModel):
    name: str
    display_name: Optional[str]
    description: Optional[str]
    geo_type: GeoType
    is_geo_pair: Optional[str]


class PrimaryDatetime(BaseModel):
    name: str
    display_name: Optional[str]
    description: Optional[str]
    date_type: DateType
    time_format: str = Field(
        title="Time Format",
        description="The strftime formatter for this field",
        example="%y",
    )
    associated_columns: Optional[Dict]
    dateassociation: bool


class Features(BaseModel):
    name: str
    display_name: Optional[str]
    description: Optional[str]
    type: ColumnType
    feature_type: FeatureType
    units: Optional[str]
    units_description: Optional[str]


class Qualifiers(BaseModel):
    name: str
    display_name: Optional[str]
    description: Optional[str]
    type: ColumnType
    feature_type: FeatureType
    units: Optional[str]
    units_description: Optional[str]
    qualifies: List[str]


class Meta(BaseModel):
    ftype: FileType


class DatasetAnnotation(BaseModel):
    date: List[PrimaryDatetime]
    geo: List[PrimaryGeography]
    features: List[Features]
    qualifiers: Optional[List[Qualifiers]]
    meta: Optional[Meta]
