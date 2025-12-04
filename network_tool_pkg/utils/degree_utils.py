import networkx as nx

# -------------------- degree sequence 생성 함수 : 랜덤 모델 생성 시 필요한 degree sequence를 생성하는 함수 --------------------
def create_degree_sequence(G) :

  # ---------- 네트워크 타입 확인 (예외 처리) ----------

  if not isinstance(G, nx.Graph) :
    raise TypeError('입력한 네트워크의 형태가 올바르지 않습니다. network_pkg.utils.preprocessing.preprocess_network()로 전처리를 먼저 실행하십시오.')

  if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)) :
    raise TypeError('입력한 네트워크는 multi-edge가 존재 가능한 MultiGraph 형식입니다. network_pkg.utils.preprocessing.preprocess_network()로 전처리를 먼저 실행하십시오.')

  if len(G.nodes()) == 0 :
    raise ValueError('입력한 네트워크는 빈 그래프입니다. network_pkg.utils.preprocessing.preprocess_network()로 전처리를 먼저 실행하십시오.')

  if len(G.edges()) == 0 :
    raise ValueError('입력한 네트워크는 엣지가 존재하지 않습니다. network_pkg.utils.preprocessing.preprocess_network()로 전처리를 먼저 실행하십시오.')
  
  self_loop_count = 0

  for node1, node2 in G.edges() :

    if node1 == node2 :
      self_loop_count += 1
      break

  if self_loop_count > 0 :
    raise ValueError('입력한 네트워크는 self-loop가 존재합니다. network_pkg.utils.preprocessing.preprocess_network()로 전처리를 먼저 실행하십시오.')

  # ---------- degree sequence 생성 ----------

  degree_sequence = [d for _, d in G.degree()]

  # ---------- 유효한 degree 값 확인 (예외 처리) ----------

  if 0 in degree_sequence :
    raise ValueError('입력한 네트워크는 isolated node가 존재합니다. network_pkg.utils.preprocessing.preprocess_network()로 전처리를 먼저 실행하십시오.')

  return degree_sequence





# -------------------- stub 전처리 함수 : 랜덤 모델 생성 시 degree sequence의 영향을 최소화하여 stub의 합을 올바른 형태로 변경해주는 함수 --------------------
def preprocess_stub(degree_sequence) :

  # ---------- degree sequence 타입 확인 (예외 처리) ----------

  if not isinstance(degree_sequence, list) :
    raise TypeError('입력한 degree sequence의 형태가 올바르지 않습니다. list 형태로 입력하십시오.')

  if len(degree_sequence) == 0 :
    raise ValueError('입력한 degree sequence는 빈 list 입니다. preprocessing.preprocess_network()로 전처리를 먼저 실행하십시오.')

  for degree in degree_sequence :

    if not isinstance(degree, int) :
      raise TypeError('입력한 degree sequence에는 정수 형태만 포함되어야합니다. 올바른 네트워크를 preprocessing.preprocess_network()로 전처리하십시오.')
    
    if degree < 0 :
      raise ValueError('입력한 degree sequence에 음수 degree가 포함되어 있습니다. 올바른 네트워크를 preprocessing.preprocess_network()로 전처리하십시오.')
    
  # ---------- stub 총합이 짝수인 경우  ----------

  degree_sum = sum(degree_sequence)

  if degree_sum % 2 == 0 :
    print('[stub prepocessing] stub의 총합이 짝수입니다. 수정이 불필요합니다.')

    return degree_sequence

  # ---------- stub 총합이 홀수인 경우  ----------
  
  print('[stub prepocessing] stub의 총합이 홀수입니다. degree sequence의 최소한의 수정을 실행합니다.')
  
  try :
    min_degree = min(degree_sequence)
    min_idx = degree_sequence.index(min_degree)

    degree_sequence[min_idx] += 1
    print('[stub prepocessing] 보정 완료 : 최소 차수({})를 1 증가시켜 {}로 수정하였습니다. 보정된 차수 합 : {}'.format(min_degree, min_degree+1, sum(degree_sequence)))

  except ValueError :
    raise ValueError('입력한 degree sequence가 잘못된 형태입니다. 올바른 네트워크를 preprocessing.preprocess_network()로 전처리하십시오.')

  return degree_sequence

