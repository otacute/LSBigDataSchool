# Ctrl + Enter
# Shift + 화살표 : 블록

a = 1
a

# 파워쉘 명령어 리스트
# ls : 파일 목록
# cd : 폴더 이동
# . 현재폴더
# .. 상위폴더

# Show folder in new window :
# 해당위치 탐색기

# Tab\Shift Tab : 자동완성\옵션 변경
# cls : 화면 정리

a = 10
a

a = "안녕하세요!"
a

a = '"안녕하세요~"라고 아빠가 말했다.'
a

# 궁금
a = ' "\'안녕하세요~\'"라고 아빠가 말했다. '
a

a = [1,2,3]
a

b = [4,5,6]
b

a+b

a = "안녕하세요!"
a

b = 'LS 빅데이터 스쿨!'
b

a + ' ' + b

print(a)

num1 = 3
num2 = 5
num1 + num2

# 카멜 케이스 : LsBigdataSchool (class 선언 시)
# 스네이크 케이스 : ls_bigdata_school (변수 선언 시)

a = 10
b = 3.3

print("a + b =", a + b)
print("a - b =", a - b)
print("a * b =", a * b)
print("a / b =", a / b)
print("a % b =", a % b) # 나머지
print("a // b =", a // b) # 몫
print("a ** b =", a ** b) # 거듭제곱
print("a + b =", a + b)

# Shift + Alt + 아래 화살표 : 아래로 복사
# Ctrl + Alt + 아래화살표 : 커서 여러개 선택
(a * 2) // 7
(a * 2) // 7
(a * 2) % 7
(a * 2) % 7

a == b
a != b
a < b 
a > b
a <= b
a >= b


# 2에 4승과 12453을 7로 나눈 몫을 더해서 8로 나눴을 때 나머지

a = ( (2**4) + (12453//7) ) % 8

# 9의 7승을 12로 나누고, 36452를 253로 나눈 나머지에 곱한 수 중 큰 것은?

b = (9**7) / 12 
c = 36452 % 253
d = b * c

print(a)
print(d)

print(a<d)

user_age = 25
is_adult = user_age >= 18
print("성인입니까?", is_adult)

# True, False는 예약어
False 
True

TRUE = 1
a = "True"
b = TRUE 
c = true
# 현재 true를 변수로 인식하고 있는데, 변수를 지정한 적이 없어서 Error 발생
d = True

# True, False
a = True
b = False

a and b
a or b
not a

# True : 1
# False : 0

True + True
True + False
False + False

# and 연산자
True and False
True and True
False and False
False and True

# and는 곱셈으로 치환 가능
True  * False
True  * True
False * False
False * True

# or 연산자
True or True
True or False
False or True
False or False

True  or True
True  or False
False or True
False or False

a = True
b = False
a or b
min(a + b, 1)

a = 3
# a = a + 10
a += 10
a

a -= 4

a %= 3
a += 12
a **= 2
a //= 2

str1 = "hello"
str1 + str1
str1 * 3 

# 문자열 변수 할당
str1 = "Hello! "
# 문자열 반복
repeated_str = str1 * 3
print("Repeated string:", repeated_str)

str1 * -2

# 정수 : int(eger)
# 실수 : float (double)

# 단항 연산자
x = 5
+x
-x
~x

bin(5)
bin(-6)

x = 3 
~x
bin(~x)

bin(-3)
bin(-16)
~15
bin(~15)

bin(0)
~0

bin(-2)

bin(0)
bin(~0)

max(3,4)
var1 = [1,2,3]
sum(var1)

! pip install pydataset
import pydataset
pydataset.data()

df = pydataset.data('HairEyeColor')
df

! pip install pandas
! python.exe -m pip install --upgrade pip

#테스트 입니다~

! pip install jupyter
! pip install pyyaml
