import datetime
#Assuming this will be part of the run.py as a helper function

def get_post_times(start_date):
#Given the starting day (day 1) in DD/MM/YYYY, returns a list of timestamps
    start_date = start_date.split("/")

    start_ts = datetime.datetime(int(start_date[2]),int(start_date[0]), int(start_date[1])).timestamp()
    timestamps = []

    #-7h 15m from start time = -26100
    timestamps.append(start_ts - 26100)

    #Day 1 intro: +9h to start time = 32400
    day_1_intro = start_ts + 32400
    timestamps.append(day_1_intro)

    timestamps = add_five_posts(timestamps,day_1_intro)

    #Day 2 intro: 24h to day_1_intro = 86400
    day_2_intro = day_1_intro + 86400
    timestamps.append(day_2_intro)

    timestamps = add_five_posts(timestamps,day_2_intro)

    #Day 3 intro: 24h to day_2_intro = 86400
    day_3_intro = day_2_intro + 86400
    timestamps.append(day_3_intro)
    timestamps = add_five_posts(timestamps,day_3_intro)

    return timestamps

def add_five_posts(timestamps, start_time):
    #Given a 9am intro time, add 1 hour to get to 10am
    new_time = start_time + 3600
    timestamps.append(new_time)
    #Given a 10am time, adds 2h 3 times representing 10-12,12-2,2-4
    for x in range(3):
        new_time = new_time + 7200
        timestamps.append(new_time)
    #Given a 4pm time, add 45m representing end time
    timestamps.append((new_time + 2700))

    return timestamps

#Testing
times = get_post_times("02/10/1989")
print(times)
