"""
GPU 기반 치료 시간 계산기
"""
import cupy as cp
import numpy as np
import logging
import numba
from typing import List, Dict, Tuple, Any, Optional, Union

# 로컬 모듈 임포트
from config import Config
from core.gpu_utils import GPUMemoryManager
from data_models.calculation_data import GatingPeriod, TreatmentTimeResult

logger = logging.getLogger(__name__)


@numba.njit(cache=True)
def _calculate_time_simulation_numba(
    n_layers: int,
    scan_times_np: np.ndarray,
    layer_switching_time: float,
    t_on: float,
    t_total: float
) -> float:
    """
    Numba JIT-compiled function for time simulation.
    Calculates the total time required to deliver all layers considering gating cycles.
    """
    current_t = 0.0
    epsilon = 1e-6  # For float comparisons

    for i in range(n_layers):
        remaining_beam_on_for_layer = scan_times_np[i]

        # Inner loop: process the current layer until its beam_on_time is delivered
        while remaining_beam_on_for_layer > epsilon:
            cycle_pos = current_t % t_total

            # Adjust cycle_pos if it's very close to t_total (mimicking original logic)
            # if np.isclose(cycle_pos, t_total): # np.isclose is not numba compatible
            if abs(t_total - cycle_pos) < 1e-4 : # Check if cycle_pos is very close to t_total
                 cycle_pos = 0.0

            # Check if currently in the ON period of the gating cycle
            if (t_on - cycle_pos) > epsilon:  # Currently in ON period
                time_available_in_cycle = t_on - cycle_pos
                time_to_use_this_step = min(time_available_in_cycle, remaining_beam_on_for_layer)

                remaining_beam_on_for_layer -= time_to_use_this_step
                current_t += time_to_use_this_step
            else:  # Currently in OFF period
                # Jump to the beginning of the next ON period
                current_t += (t_total - cycle_pos)
                # No beam_on_time is delivered during the OFF period

        # Add layer switching time after completing a layer (if not the last layer)
        if i < n_layers - 1:
            current_t += layer_switching_time

    return current_t


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
        
        # 추가: T_total 유효성 검사 (Numba 함수에서 0으로 나누기 방지)
        if T_total <= 1e-9: # T_total이 0 또는 매우 작은 경우 (epsilon 비교)
            logger.error(f"T_total ({T_total})이 0이거나 매우 작아 시뮬레이션이 불가능합니다. "
                         "게이팅 주기를 확인하세요. 폴백 결과를 반환합니다.")
            # scan_times는 CuPy 배열이므로 cp.sum 사용
            fallback_total_time = float(cp.sum(scan_times)) + (layer_switching_time * (n_layers -1) if n_layers > 0 else 0)
            return {
                'total_time': fallback_total_time,
                'beam_on_time': float(cp.sum(scan_times)),
                'efficiency': 1.0 if fallback_total_time > 0 and float(cp.sum(scan_times)) > 0 else 0.0
            }

        # Numba 함수 호출을 위한 준비
        # scan_times (CuPy 배열)를 NumPy 배열로 변환
        scan_times_np = cp.asnumpy(scan_times)

        # Numba 헬퍼 함수 호출
        total_simulated_time = _calculate_time_simulation_numba(
            n_layers,
            scan_times_np,
            layer_switching_time,
            T_on,
            T_total
        )
        
        # 총 빔 조사 시간 계산 (원본 CuPy 배열 사용)
        total_beam_on_time = float(cp.sum(scan_times))
        
        # 효율성 계산
        efficiency = total_beam_on_time / total_simulated_time if total_simulated_time > 0 else 0.0
        
        # 결과 반환
        return {
            'total_time': total_simulated_time, # Numba 함수에서 계산된 float 값
            'beam_on_time': total_beam_on_time, # float로 변환된 값
            'efficiency': efficiency # float 값
        }