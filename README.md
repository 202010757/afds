# LG AIMERS 7기 프로젝트

## 프로젝트 개요
영업장별/메뉴별 매출수량 예측을 위해 시계열 특성과 머신러닝, 딥러닝을 결합한 풀스택 예측 파이프라인을 구현했습니다. LightGBM과 LSTM을 활용한 앙상블 모델로, 다양한 Feature Engineering과 Optuna 기반 하이퍼파라미터 튜닝을 적용하여 예측 성능을 극대화합니다.

## 주요 기능 및 흐름

1. **데이터 전처리 및 Feature Engineering**
	- 영업일자, 요일, 월, 주말여부 등 날짜 기반 파생변수 생성
	- 이동평균, 표준편차, 최소/최대값, 증감률 등 시계열 통계 Feature 추가
	- LabelEncoder로 메뉴별 ID 변환

2. **Supervised Dataset 생성**
	- LOOKBACK(28일) 시계열 lag feature와 추가 변수, PREDICT(7일) 예측값을 포함한 학습 데이터 생성

3. **LightGBM 모델링 및 Optuna 튜닝**
	- Optuna로 num_leaves, learning_rate, n_estimators 등 하이퍼파라미터 최적화
	- 각 예측 step별로 LGBM 모델 학습 및 저장

4. **LSTM 딥러닝 모델링 및 튜닝**
	- PyTorch 기반 MultiOutputLSTM 구현 (7일 멀티타깃)
	- MinMaxScaler로 정규화, Optuna로 hidden_dim, num_layers, dropout, lr 튜닝
	- 전체 데이터로 최종 학습

5. **앙상블 예측 및 결과 생성**
	- 테스트셋별로 LGBM/LSTM 예측값 평균 앙상블
	- sample_submission 포맷에 맞게 결과 변환 및 저장

## 사용 기술
- Python, Pandas, Numpy, Scikit-learn, LightGBM, PyTorch, Optuna, TQDM

## 코드 구조
- `fullstack_sales_prediction.py`: 전체 파이프라인 코드 (데이터 처리, 모델링, 예측, 제출)

## 실행 방법
1. `train/train.csv`, `test/TEST_*.csv`, `sample_submission.csv` 파일 준비
2. `fullstack_sales_prediction.py` 실행
3. `submission_fullstack.csv` 결과 파일 생성

## 주요 코드 하이라이트
- Feature Engineering 함수: 다양한 시계열 통계 feature 자동 생성
- Optuna 기반 LGBM/LSTM 튜닝: 빠르고 효율적인 하이퍼파라미터 탐색
- MultiOutputLSTM: 7일 멀티타깃 예측 구조
- 앙상블 전략: 머신러닝+딥러닝 결합으로 예측 안정성 향상

## 포인트
- 실전 시계열 예측 문제에 대한 end-to-end 파이프라인 설계 경험
- 머신러닝/딥러닝/앙상블/튜닝 등 다양한 기법 통합
- 코드의 확장성, 재현성, 실용성 강조