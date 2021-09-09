from apscheduler.schedulers.blocking import BlockingScheduler
# from datetime import datetime


def scheduler(my_job, run_datetime, args_=None):
    """

    :param args_:
    :param my_job:
    :param run_datetime: datetime(2021, 8, 26, 11, 30, 5)
    :return:
    """
    sched = BlockingScheduler()
    sched.add_job(my_job, 'date', run_date=run_datetime, args=args_)
    sched.start()
