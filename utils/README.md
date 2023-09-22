### Scheduler

Scheduler wrote by apscheduler.

#### example

```python
from utils.scheduler import scheduler_trigger
def qq_fun():
    import time
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('test')


from apscheduler.triggers.cron import CronTrigger

trigger = CronTrigger(day_of_week='mon-fri', hour='14', minute='40')
scheduler_trigger(qq_fun, trigger)
```

### Send Notify

Send notify to BARK/DINGDING/WX/TG etc.

#### example

```python
from cv2box.utils.send_notify import send_notify
bark_config = {
    'BARK_PUSH': 'https://api.day.app/DxHcxxxxxRxxxxxxcm',
    # bark IP 或设备码，例：https://api.day.app/DxHcxxxxxRxxxxxxcm
    'BARK_ARCHIVE': '1',  # bark 推送是否存档
    'BARK_GROUP': 'test',  # bark 推送分组
    'BARK_SOUND': '',  # bark 推送声音
    'BARK_ICON': '',  # bark 推送图标
}
send_notify('test', 'test', bark_config)
```