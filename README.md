##### 워커 세이프 알고리즘 실행가이드

###### 0\.  소스 코드 및 데이터셋 파일 다운로드
* 소스 코드 다운로드
* Condi Guard 데이터셋 다운로드(다운로드 후 Condi_Guard 폴더에 삽입):
   https://drive.google.com/file/d/1iTvOUzx-DNoWdDCzWdreY56ijrmzjudD/view?usp=sharing
* Early Guard 데이터셋 다운로드(다운로드 후 Early_Guard 폴더에 삽입):
   https://drive.google.com/file/d/1HIOk3c4B7mJwo3uQY_k5VdA90-rKozOP/view?usp=sharing
  


###### 1\.	가상환경 생성

* 기본 코드: 
  		conda create -n \[환경이름] python=\[버전]
* 이 프로젝트를 위한 설치: 
  		conda create -n int python=3.10

###### 2\. 	가상환경 활성화 및 확인

* 활성화 기본 코드: 
  		conda activate \[환경이름]
* 예시:
  		conda activate int 
* 파이썬 버전 확인: 
  		python --version
* &nbsp;		만약 3.10버전이 아니면 가상환경 비활성화 후 새 가상환경 생성하기(1번으로 돌아감)

###### 3\. 필수 추가 명령어

* 생성된 환경 목록 확인: 
  		conda env list
* 가상환경 비활성화: 
  		conda deactivate
* 가상환경 삭제: 
  		conda remove -n \[환경이름] --all

###### 4\. 가상환경 활성화 후 실행 전 인터프리터 설정

Ctrl + Shift + P > 파이썬 인터프리터(Python: Select Interpreter) 검색 후 앞에서 생성한 가상환경 선택

###### 5\. 필요한 모듈 및 라이브러리 설치

pip install -r workersafe_requirements.txt

###### 6\. 파이썬 환경 변수 절대경로 수정
Condi Guard 폴더 내에 있는 .env 파일 열어서 절대 경로를 본인 실행 환경에 맞게 수정함
가급적 기존 코드는 주석 처리로 하고 그 밑에 본인 환경에 맞는 경로로 추가하는걸 권장함

###### 7\. 워커 세이프 알고리즘 실행
상위 폴더(INT)에서 실행(본인 환경에 맞게 코드 수정하고 이 실행가이드를 업데이트 할 것)
python workersafe.py
