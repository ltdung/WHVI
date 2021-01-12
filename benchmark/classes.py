from time import process_time


def st_time(func):
    """
    Decorator to calculate the total time of a function.

    Example:
    ```
        @st_time
        def f():
            print("Hello, world")

        f()
    ```
    """

    def st_func(*args, **kwargs):
        t1 = process_time()
        r = func(*args, **kwargs)
        t2 = process_time()
        print(f"\t{func.__name__}: {(t2 - t1):.3f} s", flush=True)
        return r

    return st_func


class Benchmark:
    def __init__(self):
        pass

    def main(self):
        # Run all methods with name benchmark_*()
        benchmark_methods = list(filter(lambda x: x.startswith('benchmark_'), dir(self)))
        for method_name in benchmark_methods:
            print(f'Evaluating {method_name}')
            eval('self.{0}()'.format(method_name))
