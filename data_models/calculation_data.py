"""
계산 데이터 구조 정의
"""
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

class BaseDataModel:
    """모든 데이터 모델의 기본 클래스"""
    
    def to_dict(self) -> Dict[str, Any]:
        """객체를 딕셔너리로 변환"""
        return {key: getattr(self, key) for key in self.__dict__ 
                if not key.startswith('_')}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseDataModel':
        """딕셔너리에서 객체 생성"""
        instance = cls()
        for key, value in data.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance

class ScanCalculationResult(BaseDataModel):
    """스캔 시간 계산 결과 클래스"""
    
    def __init__(self):
        self.mu_per_dist = None  # 거리당 MU (MU/cm)
        self.speeds = None       # 속도 배열 (cm/s)
        self.raw_scan_times = None  # 원시 스캔 시간 배열 (s)
        self.rounded_scan_times = None  # 반올림된 스캔 시간 배열 (s)
        self.layer_doserates = None  # 레이어별 선량율 (MU/s)

class TreatmentTimeResult(BaseDataModel):
    """치료 시간 계산 결과 클래스"""
    
    def __init__(self, 
               total_time: float = 0.0, 
               beam_on_time: float = 0.0, 
               efficiency: float = 0.0,
               on_beam: Optional[np.ndarray] = None):
        self.total_time = total_time      # 총 치료 시간 (s)
        self.beam_on_time = beam_on_time  # 빔 조사 시간 (s)
        self.efficiency = efficiency      # 효율성 (빔 조사 시간 / 총 치료 시간)
        self.on_beam = on_beam            # 빔 조사 상태 배열 (시각화용)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TreatmentTimeResult':
        """딕셔너리에서 TreatmentTimeResult 객체 생성"""
        return cls(
            total_time=data.get('total_time', 0.0),
            beam_on_time=data.get('beam_on_time', 0.0),
            efficiency=data.get('efficiency', 0.0),
            on_beam=data.get('on_beam', None)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """객체를 딕셔너리로 변환"""
        result = {
            'total_time': self.total_time,
            'beam_on_time': self.beam_on_time,
            'efficiency': self.efficiency
        }
        if self.on_beam is not None:
            result['on_beam'] = self.on_beam
        return result

class ParameterCombination(BaseDataModel):
    """매개변수 조합 클래스"""
    
    def __init__(self, 
               resp_period: float, 
               gating_amplitude: float, 
               layer_switching_time: float):
        self.resp_period = resp_period  # 호흡 주기 (s)
        self.gating_amplitude = gating_amplitude  # 게이팅 진폭
        self.layer_switching_time = layer_switching_time  # 레이어 전환 시간 (s)
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """객체를 튜플로 변환"""
        return (self.resp_period, self.gating_amplitude, self.layer_switching_time)
    
    @classmethod
    def from_tuple(cls, data: Tuple[float, float, float]) -> 'ParameterCombination':
        """튜플에서 ParameterCombination 객체 생성"""
        return cls(resp_period=data[0], gating_amplitude=data[1], layer_switching_time=data[2])

class BatchData(BaseDataModel):
    """배치 처리를 위한 데이터 클래스"""
    
    def __init__(self):
        self.scan_times = []  # 레이어 스캔 시간 배열
        self.parameters = []  # 매개변수 조합 배열
        self.base_gating_signals = {}  # 기본 게이팅 신호 사전 ((주기, 진폭)별)
        self.gating_periods = {}  # 게이팅 주기 정보 사전 {(주기, 진폭): GatingPeriod}
        self.time_array = None  # 시간 배열

class GatingPeriod(BaseDataModel):
    """게이팅 신호의 주기 정보 클래스"""
    
    def __init__(self, T_on: float = 0.0, T_off: float = 0.0):
        self.T_on = T_on        # 빔 on 상태 주기 (s)
        self.T_off = T_off      # 빔 off 상태 주기 (s)
        self.T_total = T_on + T_off  # 총 주기 (s)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GatingPeriod':
        """디셔너리에서 GatingPeriod 객체 생성"""
        return cls(
            T_on=data.get('T_on', 0.0),
            T_off=data.get('T_off', 0.0)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """객체를 디셔너리로 변환"""
        return {
            'T_on': self.T_on,
            'T_off': self.T_off,
            'T_total': self.T_total
        }

class BatchResult(BaseDataModel):
    """배치 처리 결과 클래스"""
    
    def __init__(self):
        self.results = {}  # 매개변수 조합별 결과 사전