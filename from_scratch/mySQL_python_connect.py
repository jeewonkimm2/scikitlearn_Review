import pandas as pd
import pymysql

# 데이터베이스와 연동, conn : 연결자
# 127.0.0.1대신 localhost 사용 가능
# password에는 mysql에서 설정한 비밀번호
conn = pymysql.connect(host='127.0.0.1', user='root', password=' ', db='test', charset='utf8')

# 데이터베이스 SQL문 실행 or 결과 받는 통로
cur = conn.cursor()
sql = "SELECT * FROM testTable;"
cur.execute(sql)

# fetchall() : 모든 데이터를 한 번에 가져올 때 사용
result = cur.fetchall()

# DataFrame 만들기
df = pd.DataFrame(result)
print(result)

# execute을 통해 테이블을 업뎃
cur.execute("INSERT INTO testTable VALUE('Weck', 25, 'Alien')")
# 업뎃 후 꼭 .commit해줘야함
conn.commit()
# 데이터 베이스 사용 후 종료
conn.close()