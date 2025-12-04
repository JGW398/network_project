# 우간다 네트워크 구조 분석 및 랜덤 모델 비교 분석

본 프로젝트는 실제 네트워크 데이터의 구조적 특이성을 분석하기 위해,
직접 구현한 네 가지 무작위 네트워크 모델 (ER, Configuration, Chung-Lu)의 앙상블 평균과
원본 데이터를 비교합니다. 모든 모델 생성 함수는 Python 클래스 형태로 관리됩니다.

---
## 1. 구현된 무작위 네트워크 모델 (random_nets Package)

모든 모델 생성 함수는 **`RandomNetGenerator`** 클래스의 메서드로 구현되었으며, 결과물은 `networkx.Graph` 객체로 반환됩니다.

| 모델 | 모델 유형 | 핵심 원리 | 특징 및 구현 (예외 처리) |
| :--- | :--- | :--- | :--- |
| **ER Model** | G(N, p) | 모든 노드 쌍이 확률 p로 독립 연결 | 확률 p의 유효성 검사 (0 <= p <= 1) 포함. |
| **Configuration Model (CM)** | G(N, k) | 원본 네트워크의 **차수 시퀀스**를 완벽하게 보존 | 차수 합이 홀수일 경우 **`ValueError`** 처리 로직 구현. |
| **Chung-Lu Model (CL)** | G(N, w) | 노드 차수 곱에 비례하여 확률적으로 연결 | 간선 확률 p_ij가 1을 초과하지 않도록 보정 구현. |

---
## 2. 환경 설정 및 사용법

### 설치 요구 사항
```bash
pip install networkx numpy matplotlib scipy
```

### 패키지 불러오기
프로젝터 폴더를 클론한 후, 다음과 같이 클래스를 불러와 사용합니다.
```python
# analysis_script.py 에서
from random_nets import RandomNetGenerator 
import networkx as nx

# 1. 원본 데이터 로드 및 차수 시퀀스 추출
G_original = nx.karate_club_graph() # 또는 외부 데이터 G를 사용
degrees = [d for _, d in G_original.degree()]
N = G_original.number_of_nodes()

# 2. 클래스 인스턴스 생성
generator = RandomNetGenerator(N_nodes=N, initial_degrees=degrees)

# 3. 모델 생성 (앙상블 생성 루프에 사용)
G_er_sample = generator.create_er_net(p=0.08)
G_cm_sample = generator.create_config_model()
```

## 핵심 분석 결과 요약
