from apscheduler.schedulers.blocking import BlockingScheduler
from disney.crawler import Crawler

sched = BlockingScheduler()

@sched.scheduled_job('interval', minutes=1)
def timed_job():
    crawler = Crawler()
    crawler.get_parks_waittime().save_dataframe().gdrive_upload().delete_local_files()

sched.start()