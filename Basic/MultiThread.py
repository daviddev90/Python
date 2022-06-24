# 멀티 스레드 : 한 문제를 여러 cpu가 나눠서 한다 

# Before
# total1 = 0

# for i in range(0, 100000000):
#     total1 += i
# print(total1)

# ----------------------------------------------------------------

# After
from threading import Thread
def calc(start , end):
    total = 0
    for i in range(start, end):
        total += i
    print(total)

if __name__ == "__main__":
    start, end = 0, 100000000
    thr1 = Thread(target=calc, args=(start, end))
    
    thr1.start()
    thr1.join()