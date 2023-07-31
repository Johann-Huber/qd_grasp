
import utils.constants as consts
import sys
import time
from datetime import timedelta


class Timer:

    def __init__(self):

        self._t_starts = {}  # dict containing the start time associated to each timer label
        self._times_in_s = {}   # dict containing the elapsed time associated to each timer label


    def start(self, label):
        self._start_sanity_checks(label)
        self._t_starts[label] = time.time()


    def stop(self, label):
        self._stop_sanity_checks(label)
        self._times_in_s[label] = time.time() - self._t_starts[label]
        self._clean_label_t_start(label)


    def get_all(self, format='seconds'):

        if not isinstance(format, str):
            raise AttributeError(f'format must be a str (given type: {type(format)})')

        if format not in consts.SUPPORTED_TIMER_FORMATS:
            raise AttributeError(f'not supported timer format: {format} (supported: {consts.SUPPORTED_TIMER_FORMATS}')

        if format == 'h:m:s':
            time_in_hms = {
                label: self._td_format(timedelta(seconds=self._times_in_s[label])) \
                for label in self._times_in_s
            }

            return time_in_hms

        return self._times_in_s

    def get_on_the_fly_time(self, label):
        self._stop_sanity_checks(label)
        curr_elapsed_time = time.time() - self._t_starts[label]
        return curr_elapsed_time

    def _clean_label_t_start(self, label):
        self._t_starts.pop(label)


    def _start_sanity_checks(self, label):
        if not isinstance(label, str):
            raise AttributeError(f'Timer label must be a str (type = {type(label)})')
        if label in self._t_starts:
            raise AttributeError(f'Double call of Timer.start() for a same label ({label})')
        if label in self._times_in_s:
            raise AttributeError(f'Label already used ({label})')


    def _stop_sanity_checks(self, label):
        if not isinstance(label, str):
            raise AttributeError(f'Timer label must be a str (type = {type(label)})')
        if label not in self._t_starts:
            raise AttributeError(f'Timer.stop() called before Timer.start() (label={label})')

    @staticmethod
    def _td_format(td_object):
        seconds = int(td_object.total_seconds())
        periods = [
            ('year',        60*60*24*365),
            ('month',       60*60*24*30),
            ('day',         60*60*24),
            ('hour',        60*60),
            ('minute',      60),
            ('second',      1)
        ]

        strings=[]
        for period_name, period_seconds in periods:
            if seconds > period_seconds:
                period_value , seconds = divmod(seconds, period_seconds)
                has_s = 's' if period_value > 1 else ''
                strings.append("%s %s%s" % (period_value, period_name, has_s))

        return ", ".join(strings)



def timer_debug_snippet():
    timer = Timer()
    timer.start(label='fake_run')
    for i in range(100000000):
        x = i**3

    timer.stop(label='fake_run')

    times_in_s = timer.get_all(format='seconds')
    times_in_hms = timer.get_all(format='h:m:s')

    print('times_in_s =\n', times_in_s)
    print('times_in_hms =\n', times_in_hms)


if __name__ == '__main__':
    sys.exit(timer_debug_snippet())

