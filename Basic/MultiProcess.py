# 멀티프로세스(멀티프로그래밍) : 프로세스를 여러개 사용 , 한 프로세스당 한 문제씩 할당해 줌

# Before
# total1 = total2 = total3 = 0

# for i in range(0, 100000000):
#     total1 += i
# print(total1)

# for i in range(100000001, 200000000):
#     total2 += i
# print(total2)

# for i in range(200000001, 300000000):
#     total3 += i
# print(total3)
# print(">>End<<")

# ----------------------------------------------------------------

# After
from multiprocessing import Process
def start_get(start ,end):
    total = 0
    for i in range(start , end):
        total += i
    print(total)

if __name__ == "__main__":
    # 프로세스를 생성 합니다. , 각자 프로세스를 만들어 준다 / M1 8개까지 가능
    p0 = Process(target=start_get, args=(0,100000000))
    p1 = Process(target=start_get, args=(100000001, 200000000))
    p2 = Process(target=start_get, args=(200000001, 300000000))

    # 프로세스의 시작
    p0.start()
    p1.start()
    p2.start()

    # 순서를 정리한다 : 다른 프로세스가 종료될 때까지 대기한다 , 출력순서 . 꼭 써줘야 한다@
    p0.join()
    p1.join()
    p2.join()

    print(">>>> End <<<<")
