import datetime

def get_readable_timestamp(stamp):
    return datetime.datetime.fromtimestamp(
                    stamp
                ).strftime("%H:%M:%S.%f")[:-3]