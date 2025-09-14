import scrapy
from scrapy.loader import ItemLoader
from trademe.items import TrademeItem
import pendulum

class TrademeCrawler(scrapy.Spider):
    name = 'trademe'
    start_urls = ['https://www.trademe.co.nz/a/motors/cars'] #/mitsubishi/asx/search?sort_order=motorspriceasc&user_region=2&price_max=40000&odometer_max=90000

    def parse(self, response):
        
        car_page_links = response.css('.o-card a')
        yield from response.follow_all(car_page_links, self.parse_author)

        for a in response.css('ul.o-pagination__nav-list a'):
            yield response.follow(a, callback=self.parse)


    def parse_author(self, response):
        l = ItemLoader(item=TrademeItem(), response=response)

        l.add_value('url', response.url)
        l.add_css('name', ".tm-motors-listing__title::text")

        l.add_css('price', ".tm-motors-pricing-box__price::text"),
        l.add_css('model_type', ".tm-motors-listing-title__model-detail::text"),
        l.add_css('period', ".tm-motors-date-city-watchlist__date::text"),

        l.add_css('location', ".tm-motors-date-city-watchlist__location::text"),
        l.add_css('views', ".tm-motors-date-city-watchlist__views-container strong::text"),
        l.add_css('watchlist', ".tm-motors-date-city-watchlist__watchlists strong::text"),
        l.add_css('description', "div.tm-markdown p::text"),

        l.add_css('vehicle_info_meta', ".tm-motors-vehicle-attributes__tag--content title::text"),
        l.add_css('vehicle_info_meta2', ".tm-motors-vehicle-attributes__attribute-name::text"),
        

        l.add_css('vehicle_info', ".tm-motors-vehicle-attributes__tag--content::text"),

        l.add_css('ratings_fuel', ".tm-motors-listing-ratings__fuel::text"),
        l.add_css('ratings_carbon_emissions', ".tm-motors-listing-ratings__carbon::text"),
        l.add_css('background_item_check', ".tm-background-check__background-check-item-title::text"),
        l.add_css('background_value_check', ".tm-background-check__status::text"),
        l.add_css('vehicle_features', ".tm-motors-listing-features__item::text"),
            
        l.add_css('price_details', ".tm-clean-vehicle-and-orc-information__container div::text"),
        l.add_css('seller', ".tm-dealer-info__dealer-links-website::text"),
        
        l.add_value("datetime", pendulum.now().format("YYYY-MM-DD"))
        return l.load_item()