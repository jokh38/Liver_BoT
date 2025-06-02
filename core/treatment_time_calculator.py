"""
GPU 기반 치료 시간 계산기
"""
import cupy as cp
import numpy as np
import logging
from typing import List, Dict, Tuple, Any, Optional, Union

# 로컬 모듈 임포트
from config import Config
from core.gpu_utils import GPUMemoryManager
from data_models.calculation_data import GatingPeriod, TreatmentTimeResult

logger = logging.getLogger(__name__)

class TreatmentTimeCalculator:
    """치료 시간 계산 전용 클래스"""
    
    def __init__(self, time_resolution: float = Config.TIME_RESOLUTION):
        """
        치료 시간 계산기 초기화
        
        Args:
            time_resolution: 시간 해상도 (s)
        """
        self.time_resolution = time_resolution
        self.gpu_manager = GPUMemoryManager()
        logger.info("치료 시간 계산기 초기화 완료")
    
    def calculate_treatment_time(self, 
                              scan_times: np.ndarray, 
                              layer_switching_time: float,
                              gating_periods: Optional[GatingPeriod] = None, 
                              resp_period: Optional[float] = None, 
                              gating_amplitude: Optional[float] = None) -> TreatmentTimeResult:
        """
        치료 시간 계산
        
        Args:
            scan_times: 레이어 스캔 시간 배열
            layer_switching_time: 레이어 전환 시간
            gating_periods: 게이팅 주기 정보 객체 (선택적)
            resp_period: 호흡 주기 (초, 선택적)
            gating_amplitude: 게이팅 진폭 (선택적)
            
        Returns:
            치료 시간 계산 결과 객체
        """
        # GPU로 배열 전송
        scan_times_gpu = self.gpu_manager.array_to_gpu(scan_times)
        
        if scan_times_gpu is None:
            logger.error("GPU 배열 변환 실패")
            return TreatmentTimeResult()
        
        result_dict = self._calculate_treatment_time_gpu(
            scan_times=scan_times_gpu,
            layer_switching_time=layer_switching_time,
            gating_periods=gating_periods,
            resp_period=resp_period,
            gating_amplitude=gating_amplitude
        )
        
        # 결과 객체 생성
        treatment_result = TreatmentTimeResult(
            total_time=result_dict['total_time'],
            beam_on_time=result_dict['beam_on_time'],
            efficiency=result_dict['efficiency']
        )
        
        return treatment_result
    
    def _calculate_treatment_time_gpu(self, 
                                  scan_times: cp.ndarray, 
                                  layer_switching_time: float,
                                  gating_periods: Optional[GatingPeriod] = None, 
                                  resp_period: Optional[float] = None, 
                                  gating_amplitude: Optional[float] = None) -> Dict[str, float]:
        """
        GPU에서 치료 시간 계산 (새로운 알고리즘 적용)
        
        Args:
            scan_times: 레이어 스캔 시간 배열 (GPU)
            layer_switching_time: 레이어 전환 시간
            gating_periods: 게이팅 주기 정보 객체 (선택적)
            resp_period: 호흡 주기 (초, 선택적)
            gating_amplitude: 게이팅 진폭 (선택적)
            
        Returns:
            계산 결과 딕셔너리
        """
        # 입력 유효성 검사
        if len(scan_times) == 0:
            return {
                'total_time': 0.0,
                'beam_on_time': 0.0,
                'efficiency': 0.0
            }
        
        # 총 레이어 수
        n_layers = len(scan_times)
        
        # 1. 주기 정보 (T_on, T_off, T_total) 설정
        if gating_periods is not None:
            # 게이팅 주기 정보 사용
            T_on = gating_periods.T_on
            T_total = gating_periods.T_total

        else:
            # gating_periods가 None인 경우, 필요한 정보가 있는지 확인
            if resp_period is None or gating_amplitude is None:
                logger.warning("게이팅 주기 정보 및 호흡 주기/진폭 정보 없음. 계산 불가.")
                return {
                    'total_time': float(cp.sum(scan_times)),
                    'beam_on_time': float(cp.sum(scan_times)),
                    'efficiency': 1.0
                }
            
            # 호흡 주기와 게이팅 진폭으로 게이팅 주기 정보 계산이 필요하지만,
            # 이 메서드에서는 외부에서 생성된 gating_periods를 사용해야 함
            logger.warning("외부에서 생성된 게이팅 주기 정보가 필요합니다.")
            return {
                'total_time': float(cp.sum(scan_times)),
                'beam_on_time': float(cp.sum(scan_times)),
                'efficiency': 1.0
            }
        
        # 2. 초기 시뮬레이션 시간 t = 0 설정
        t = 0.0
        
        # 3. 각 레이어에 대해 시뮬레이션 수행
        for i in range(n_layers):
            # a. 현재 레이어의 필요 빔 조사 시간
            remaining = float(scan_times[i])
            # logger.info("레이어 {}의 필요 빔 조사 시간: {:.6f}초 / T_on: {:.6f}초 / T_total: {:.6f}초".format(i, remaining, T_on, T_total))
            
            # b. 레이어 빔 조사가 완료될 때까지 시뮬레이션
            while cp.round(remaining, 5) > 0:
                # i. 현재 시간 t가 게이팅 주기 내 어느 위치인지 계산
                current_cycle_position = cp.round(t % T_total, 5)
                if T_total - current_cycle_position < 1e-4:
                    current_cycle_position = 0.0
                
                # logger.info(f"현재 시간: {t:.5f}초, 게이팅 주기 내 위치: {current_cycle_position:.5f}초, 남은 필요 시간: {remaining:.5f}초")

                # ii. 현재 주기 내 남아있는 ON 시간 계산
                if cp.round(T_on - current_cycle_position, 5) > 0:
                    # ON 구간에 있는 경우
                    available_time = T_on - current_cycle_position
                    
                    # 사용 가능한 시간과 남은 필요 시간 중 작은 값 사용
                    used_time = min(available_time, remaining)
                    
                    # 남은 필요 시간 감소
                    remaining -= used_time

                    # 시간 증가
                    t += used_time

                    # logger.info("T_curr: {:.5f}초 / T_now: {:.5f}초 / T_avail: {:.5f}초 / T_used: {:.5f}초 / T_rem: {:.5f}초".format(current_cycle_position, t, available_time, used_time, remaining))
                else:
                    # OFF 구간에 있는 경우, 다음 ON 구간으로 점프
                    next_on_time = t + (T_total - current_cycle_position)
                    t = next_on_time
                    # logger.info(f"gating off time is added. t: {t:.5f}초")
            
            # logger.info("레이어 {} 완료 후 레이어 전환 시간 추가: {:.2f}초".format(i, layer_switching_time))
            # c. 레이어 완료 후 레이어 전환 시간 추가
            if i < n_layers - 1:  # 마지막 레이어가 아닌 경우에만
                t += layer_switching_time
        
        # logger.info("시뮬레이션 완료")
        
        # 4. 총 시간 = 마지막 t
        total_time = t
        
        # 5. 총 빔 조사 시간 계산 (스캔 시간 합)
        total_beam_on_time = float(cp.sum(scan_times))
        
        # 6. 효율성 = 총 빔 조사 시간 / 총 시간
        efficiency = total_beam_on_time / total_time if total_time > 0 else 0.0
        
        # 값 타입 변환하여 반환
        return {
            'total_time': float(total_time),
            'beam_on_time': float(total_beam_on_time),
            'efficiency': float(efficiency)
        }