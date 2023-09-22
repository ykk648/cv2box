# from datetime import datetime


def scheduler_onetime(my_job, run_datetime, args_=None):
    """

    :param args_:
    :param my_job:
    :param run_datetime: datetime(2021, 8, 26, 11, 30, 5)
    :return:
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    sched = BlockingScheduler(timezone='Asia/Shanghai')
    sched.add_job(my_job, 'date', run_date=run_datetime, args=args_)
    sched.start()


def scheduler_repeat(my_job, hour, minute, args_=None):
    """

    :param args_:
    :param my_job:
    :param run_datetime: datetime(2021, 8, 26, 11, 30, 5)
    :return:
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    sched = BlockingScheduler(timezone='Asia/Shanghai')
    sched.add_job(my_job, 'cron', hour=hour, minute=minute, args=args_)
    sched.start()


def scheduler_trigger(my_job, trigger):
    from apscheduler.schedulers.blocking import BlockingScheduler
    sched = BlockingScheduler(timezone='Asia/Shanghai')
    scheduler = BlockingScheduler()

    scheduler.add_job(my_job, trigger=trigger)
    scheduler.start()
