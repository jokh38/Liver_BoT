"""
DICOM 구조를 위한 데이터 클래스 정의
"""
import numpy as np
from typing import Optional, List, Dict, Tuple

class LineSegment:
    """개별 라인 세그먼트 정보"""
    __slots__ = ('position', 'weight', 'distance', 'dose_rate', 
                'raw_scan_time', 'rounded_scan_time', 'mu_per_dist', 
                'speed', 'energy')
    
    def __init__(self, position: Optional[np.ndarray] = None, 
                weight: float = 0.0, distance: float = 0.0) -> None:
        self.position = position  # 위치 좌표 (x, y)
        self.weight = weight      # 가중치 (MU)
        self.distance = distance  # 다음 세그먼트까지의 거리 (cm)
        
        # 계산된 값들
        self.dose_rate = 0.0      # 선량율 (MU/s)
        self.raw_scan_time = 0.0  # 실제 스캔 시간 (초)
        self.rounded_scan_time = 0.0  # 반올림된 스캔 시간 (초)
        self.mu_per_dist = 0.0    # 거리당 MU (MU/cm)
        self.speed = 0.0          # 속도 (cm/s)
        self.energy = 0.0         # 에너지

class Layer:
    """레이어 정보"""
    __slots__ = ('energy', 'cum_weight_now', 'cum_weight_next', 'positions', 
                'weights', 'mlc_positions', 'num_positions', 'tune_id', 'line_segments', 
                'total_mu', 'total_scan_time', 'layer_doserate', 'dynamic_range')
    
    def __init__(self, energy: float = 0.0, cum_weight_now: float = 0.0, 
                cum_weight_next: float = 0.0, positions: Optional[np.ndarray] = None, 
                weights: Optional[np.ndarray] = None, mlc_positions: Optional[np.ndarray] = None, 
                num_positions: str = "", tune_id: str = "") -> None:
        self.energy = energy                # 에너지 (MeV)
        self.cum_weight_now = cum_weight_now    # 현재 누적 가중치
        self.cum_weight_next = cum_weight_next  # 다음 누적 가중치
        self.positions = positions          # 위치 배열 (n, 2)
        self.weights = weights              # 가중치 배열 (n, 1)
        self.mlc_positions = mlc_positions  # MLC positions (46, 2)
        self.num_positions = num_positions  # 포지션 수 정보
        self.tune_id = tune_id              # 튜닝 ID
        
        # 계산될 값들
        self.line_segments = []             # 라인 세그먼트 객체 리스트
        self.total_mu = 0.0                 # 총 MU
        self.total_scan_time = 0.0          # 총 스캔 시간
        self.layer_doserate = 0.0           # 레이어 선량율
        self.dynamic_range = 0.0            # 동적 범위 

        # 라인 세그먼트 생성
        self._create_line_segments()
        
    def _create_line_segments(self) -> None:
        """포지션과 가중치 정보로부터 라인 세그먼트 객체 생성"""
        if self.positions is None or self.weights is None:
            return
            
        num_segments = self.positions.shape[0]
        self.line_segments = [LineSegment() for _ in range(num_segments)]
        
        # 첫 번째 세그먼트 초기화
        self.line_segments[0].position = self.positions[0]
        self.line_segments[0].weight = self.weights[0] if len(self.weights) > 0 else 0.0
        self.line_segments[0].energy = self.energy
        
        # 나머지 세그먼트 초기화
        if num_segments > 1:
            # 벡터화된 거리 계산
            diff_vectors = np.diff(self.positions, axis=0)
            distances = np.sqrt(np.sum(diff_vectors**2, axis=1))
            
            for i in range(1, num_segments):
                self.line_segments[i].position = self.positions[i]
                self.line_segments[i].weight = self.weights[i] if i < len(self.weights) else 0.0
                self.line_segments[i].distance = distances[i-1]
                self.line_segments[i].energy = self.energy
            
            # 임시 변수 해제
            del diff_vectors
        
        # Dynamic range 계산
        non_zero_weights = self.weights[self.weights > 0]  # 0이 아닌 값만 필터링
        if len(non_zero_weights) > 0:  # 0이 아닌 값이 존재할 때만 계산
            self.dynamic_range = np.max(self.weights) / np.min(non_zero_weights)
        else:
            self.dynamic_range = 0.0  # 0이 아닌 값이 없을 경우 기본값 설정

        # 총 MU 계산
        self.total_mu = np.round(self.cum_weight_next - self.cum_weight_now, 3)

class Port:
    """포트(빔) 정보를 저장하는 클래스"""
    __slots__ = ('layers', 'total_scan_time', 'beam_name', 'aperture', 'mlc_y')
    
    def __init__(self) -> None:
        self.layers = []           # 레이어 객체 리스트
        self.total_scan_time = 0.0  # 총 스캔 시간
        self.beam_name = []         # beam name
        self.aperture = []         # block shape
        self.mlc_y = []            # mlc y positions
        
    def get_layers(self) -> List[Layer]:
        """레이어 목록 반환"""
        return self.layers
