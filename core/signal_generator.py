"""
호흡 주기 기반 게이팅 신호 생성 모듈

이 모듈은 호흡 주기와 게이팅 진폭을 입력으로 받아 수학적 모델을 통해
게이팅 주기 정보(T_on, T_off)를 직접 계산하는 기능을 제공합니다.

주요 기능:
- calculate_gating_periods: 호흡 주기와 게이팅 진폭으로부터 직접 T_on, T_off 계산
- generate_gating_signal: 호흡 신호 또는 주기/진폭으로부터 게이팅 신호 생성
"""
import numpy as np
import logging

# 로컬 모듈 임포트
from config import Config
from data_models.calculation_data import GatingPeriod

# 로깅 설정
logger = logging.getLogger(__name__)

class SignalGenerator:
    """게이팅 신호 생성 및 분석 클래스"""
    
    def __init__(self, time_resolution: float = Config.TIME_RESOLUTION):
        """
        게이팅 신호 생성기 초기화
        
        Args:
            time_resolution: 시간 해상도 (초)
        """
        self.time_resolution = time_resolution
        logger.info(f"SignalGenerator 초기화 완료 (시간 해상도: {time_resolution}초)")
    
    def generate_gating_signal(self, gating_amplitude: float = None,
                             resp_period: float = None) -> GatingPeriod:
        """호흡 신호에서 게이팅 신호 생성
        
        Args:
            resp_signal: 호흡 신호 배열 (이전 인터페이스 호환용)
            gating_amplitude: 게이팅 임계값 진폭
            time_array: 시간 배열 (새 인터페이스용)
            resp_period: 호흡 주기 (초, 새 인터페이스용)
            
        Returns:
            게이팅 신호 배열 (0 또는 1) 또는 (게이팅 신호, 게이팅 주기 정보) 튜플
        """
        # 시간 배열 생성
        time_array = np.arange(0, resp_period, self.time_resolution)

        # 주기와 진폭으로 신호 생성
        if resp_period is not None and gating_amplitude is not None:
            
            # 임계값 계산
            threshold = gating_amplitude
            
            # 호흡 신호 생성 
            phase_radians = 2*np.pi*time_array/resp_period - np.pi/4
            resp_signal_nn = np.sin(phase_radians) + np.sin(phase_radians + 1.4/(2*np.pi))**2 + 0.5*np.sin(phase_radians + (np.pi-5.4)/2)**2
            resp_signal = resp_signal_nn/np.max(resp_signal_nn, 0)
            
            # 게이팅 신호 생성
            gating_signal = np.zeros_like(time_array)
            gating_indices = np.where(resp_signal < threshold)[0]
            gating_signal[gating_indices] = 1

            # 게이팅 주기 계산
            T_on = len(gating_indices)*self.time_resolution
            T_off = resp_period - T_on
            
            return GatingPeriod(T_on=T_on, T_off=T_off)
        
        else:
            raise ValueError("호흡 신호 또는 (시간 배열, 호흡 주기, 게이팅 진폭) 중 하나를 제공해야 합니다.")
    