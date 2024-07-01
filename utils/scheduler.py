# from datetime import datetime
from .util import try_import
blocking = try_import('apscheduler.schedulers.blocking', 'scheduler: pip install apscheduler')
BlockingScheduler = blocking.BlockingScheduler


def scheduler_onetime(my_job, run_datetime, args_=None):
    """
    :param args_:
    :param my_job:
    :param run_datetime: datetime(2021, 8, 26, 11, 30, 5)
    :return:
    """
    sched = BlockingScheduler(timezone='Asia/Shanghai')
    sched.add_job(my_job, 'date', run_date=run_datetime, args=args_, max_instances=1, misfire_grace_time=1)
    sched.start()
    sched.shutdown(wait=True)


def scheduler_repeat(my_job, hour, minute, args_=None):
    """

    :param args_:
    :param my_job:
    :param run_datetime: datetime(2021, 8, 26, 11, 30, 5)
    :return:
    """
    sched = BlockingScheduler(timezone='Asia/Shanghai')
    sched.add_job(my_job, 'cron', hour=hour, minute=minute, args=args_)
    sched.start()


def scheduler_trigger(my_job, trigger):
    sched = BlockingScheduler(timezone='Asia/Shanghai')
    scheduler = BlockingScheduler()

    scheduler.add_job(my_job, trigger=trigger)
    scheduler.start()
