"""
프로젝트 경로 및 설정 관리 모듈
"""
import os
import logging

# 프로젝트 루트 디렉토리 설정
# 현재 파일의 디렉토리를 기준으로 경로 설정
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 데이터 관련 경로
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')

# 데이터 파일 경로
DOSERATE_TABLE_PATH = os.path.join(ROOT_DIR, 'LS_doserate.csv')

# 로그 설정
LOG_FILE = os.path.join(LOGS_DIR, 'main.log')

# 기타 환경 설정값
class Config:
    """애플리케이션 설정 클래스"""
    
    # 디버그 모드 설정
    DEBUG_MODE = False
    
    # 선량율 관련 설정
    MIN_DOSERATE = 1.4    # (MU/s)
    MAX_SPEED = 20.0 * 100  # (cm/s)
    MIN_SPEED = 0.1 * 100   # (cm/s)
    TIME_RESOLUTION = 0.1/1000  # (s)
    
    # 시뮬레이션 관련 설정
    # TR = [0.5 + 0.5 * i for i in range(13)]  # 호흡 주기 (0.5초부터 6초까지 0.4초 단위)
    # amp_g = [0.1, 0.15, 0.2, 0.25]  # 게이팅 진폭
    TLS = [0.5*(i) for i in range(6)]  # 레이어 전환 시간 (0.5초부터 3초까지 0.5초 단위)
    DR = [10 + 20 * i for i in range(9)] # 선량률 (10MU/s부터 190MU/s까지 20MU/s 단위)
    TR = [1.01 + 0.2 * i for i in range(30)]  # 호흡 주기 (1초부터 6초까지 0.2초 단위)
    amp_g = [ 0.2]  # 게이팅 진폭
    # TLS = [2]  # 레이어 전환 시간 (0.2초부터 3초까지 0.2초 단위)
    DEFAULT_SIMULATION_DURATION = 300.0
    
    # GPU 배치 처리 관련 설정
    BATCH_SIZE = 400  # 한 번에 처리할 매개변수 조합 수
    GPU_STREAMS = 4   # 동시에 사용할 CUDA 스트림 수
    
    # 게이팅 신호 캐시 관련 설정
    GATING_CACHE_MAX_SIZE_MB = 100  # 최대 캐시 크기 (MB)
    GATING_CACHE_ENABLED = True     # 캐시 활성화 여부
    
    @classmethod
    def ensure_directories(cls):
        """필요한 디렉토리가 존재하는지 확인하고 없으면 생성"""
        for directory in [DATA_DIR, RESULTS_DIR, LOGS_DIR]:
            os.makedirs(directory, exist_ok=True)
            
        # 로그 디렉토리가 생성된 후 로그 설정
        logging.basicConfig(
            level=logging.INFO if not cls.DEBUG_MODE else logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=LOG_FILE,
            filemode='w'
        )