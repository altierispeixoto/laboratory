# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy.item import Item, Field

from itemloaders.processors import Join, MapCompose, TakeFirst, Identity
from w3lib.html import remove_tags


def clean_price(x):
    return x.replace("$",'')

def clean_location(x):
    return x.replace("Seller located in", '')

class TrademeItem(Item):

    # define the fields for your item here like:
    url =  Field(
        input_processor=MapCompose(str.title, str.lower, str.strip),
        output_processor=Join(),
    )

    name =  Field(
        input_processor=MapCompose(str.title, str.strip),
        output_processor=TakeFirst(),
    )

    price = Field(
        input_processor=MapCompose(clean_price, str.title, str.strip),
        output_processor=TakeFirst(),
    )

    model_type = Field(
        input_processor=MapCompose(str.title, str.strip),
        output_processor=TakeFirst(),
    )

    period = Field(
        input_processor=MapCompose(str.title, str.strip),
        output_processor=TakeFirst(),
    )

    location = Field(
        input_processor=MapCompose(clean_location, str.strip),
        output_processor=TakeFirst(),
    )

    views = Field(
        input_processor=MapCompose(str.title, str.strip),
        output_processor=TakeFirst(),
    )

    watchlist = Field(
        input_processor=MapCompose(str.title, str.strip),
        output_processor=TakeFirst(),
    )

    description = Field(
        input_processor=MapCompose(str.title, str.strip),
        output_processor=Join(),
    )

    vehicle_info_meta = Field(
        input_processor=MapCompose(str.title, str.strip),
        output_processor=Identity(),
    )

    vehicle_info_meta2 = Field(
        input_processor=MapCompose(str.title, str.strip),
        output_processor=Identity(),
    )

    vehicle_info = Field(
        input_processor=MapCompose(str.title, str.strip),
        output_processor=Identity(),
    )

    ratings_fuel =  Field(
        input_processor=MapCompose(str.title, str.strip),
        output_processor=Join(),
    )

    ratings_carbon_emissions = Field(
        input_processor=MapCompose(str.title, str.strip),
        output_processor=Join(),
    )


    background_item_check =  Field(
        input_processor=MapCompose(str.title, str.strip),
        output_processor=Identity(),
    )

    background_value_check =  Field(
        input_processor=MapCompose(str.title, str.strip),
        output_processor=Identity(),
    )

    vehicle_features =  Field(
        input_processor=MapCompose(str.title, str.strip),
        output_processor=Identity(),
    )

    price_details =  Field(
        input_processor=MapCompose(str.title, str.strip),
        output_processor=Identity(),
    )

    seller =  Field(
        input_processor=MapCompose(str.title, str.strip),
        output_processor=Join(),
    )

    datetime = Field(
        input_processor=Identity(),
        output_processor=TakeFirst(),
    )
    
