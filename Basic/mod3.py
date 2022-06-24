PI = 3.141592

class Math:
    # a = 1 # property
    def solve(self, r): # class에서 함수를 만들때는 self, 가 꼭 무조건 들어가야 한다. 파이썬 값이라는 표시. 부를때는 필요없음 , 정의할때만 써줌.
        return PI * (r ** 2)
    def sum(self, a, b):
        return a + b

# 주피터는 싱글cpu 터미널에서는 멀티cpu 사용 
if __name__ == "__main__":
    print(PI)
    a = Math()
    print( a.solve(2) )
    print( a.sum(PI, 4.4) )