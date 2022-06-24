def sum(a,b):
    if type(a) is not type(b) : 
        print("더하기를 할 수 없습니다.")
    else :
        return a + b


# 개발용 (분석에서는 필요없음) 
# __로 시작하는건 시스템변수
if __name__ == '__main__':
    print(sum(10,20))