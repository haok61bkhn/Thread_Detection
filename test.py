import multiprocessing as mp
import time

def _square(x):
  return x*x

def log_result(result):
  # Hàm được gọi bất kỳ khi nào _square(i) trả ra kết quả.
  # result_list được thực hiện trên main process, khong phải pool workers.
  result_list.append(result)

def apply_async_with_callback():
  pool = mp.Pool(processes=1)
  for i in range(20):
    pool.apply_async(_square, args = (i, ), callback = log_result)
  pool.close()
  pool.join()
  print(result_list)

if __name__ == '__main__':
  result_list = []
  apply_async_with_callback()