# -*- coding: utf-8 -*-
import scrapy
import re
from datetime import datetime
import pandas as pd
from naver_news.items import NaverNewsItem


class CrawlNewsSpider(scrapy.Spider):
    name = 'crawl_news'
    allowed_domains = ['news.naver.com','search.naver.com']
    url_format = "https://search.naver.com/search.naver?&where=news&query={0}&sm=tab_pge&sort=0&photo=0&field=0&reporter_article=&pd=3&ds={1}&de={1}&docid=&nso=so:r,p:from{2}to{2},a:all&mynews=0&cluster_rank=41&start=1&refresh_start=0"
    url_format = "https://search.naver.com/search.naver?&where=news&query={0}&sm=tab_pge&sort=1&photo=0&field=0&reporter_article=&pd=3&ds={1}&de={1}&docid=&nso=so:dd,p:,a:all&mynews=1&start=1&refresh_start=0"
            
    def __init__(
        self, keyword="", start="", end="", **kwargs
    ):
        startdate =  datetime.strptime(start, "%Y-%m-%d")
        enddate =  datetime.strptime(end, "%Y-%m-%d")

        self.urls= []
        for cur_date in pd.date_range(startdate, enddate):
            self.urls.append(self.url_format.format(keyword, cur_date.strftime("%Y.%m.%d"))) #, cur_date.strftime("%Y%m%d")))

    def start_requests(self):
        for url in self.urls:
            # 2227 연합인포맥스, 1001 연합뉴스, 1018 이데일리
            yield scrapy.Request(url=url, cookies={"news_office_checked": "1001,1018,2227"}, callback=self.parse)#, headers=headers)

    def parse(self, response):
        for item in response.css("ul.type01 li") :
            if item.css("a._sp_each_url") :
                url = item.css("a._sp_each_url::attr(href)").get()
                yield scrapy.Request(url, callback=self.parse_detail)
        
        next_page = response.css('a.next::attr(href)').get()
        if next_page is not None:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)
    
    def parse_detail(self, response):   
        item = NaverNewsItem()

        item['url']=response.url
        item['title']=response.css("h3#articleTitle::text").get()
        item['media']=response.css("div.press_logo img::attr(title)").get()
        item['date']=response.css("div.sponsor span.t11::text").get()
        item['content']=''.join(response.css("div#articleBodyContents::text").getall()).replace("\n","").strip()
        
        #if item['media'] in ['연합뉴스','이데일리','연합인포맥스'] : 
        yield item
