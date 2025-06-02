"""
GPU 기반 구현 모듈 - 리팩토링된 버전
"""
import logging
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

# 로컬 모듈 임포트
from config import Config, DOSERATE_TABLE_PATH
from core.scan_time_calculator import ScanTimeCalculator
from core.treatment_time_calculator import TreatmentTimeCalculator
from core.batch_processor import BatchProcessor
from core.signal_generator import SignalGenerator
from data_models.dicom_structures import Layer, Port
from data_models.calculation_data import BatchData, BatchResult, GatingPeriod, TreatmentTimeResult

# 로깅 설정
logger = logging.getLogger(__name__)

class HybridImplementation:
    """클래스 간 조율 관리 클래스 (리팩토링된 버전)"""
    
    def __init__(self, min_doserate=Config.MIN_DOSERATE, 
                max_speed=Config.MAX_SPEED, 
                min_speed=Config.MIN_SPEED,
                time_resolution=Config.TIME_RESOLUTION) -> None:
        """
        하이브리드 구현 초기화
        
        Args:
            min_doserate: 최소 선량율 (MU/s)
            max_speed: 최대 속도 (cm/s)
            min_speed: 최소 속도 (cm/s)
            time_resolution: 시간 해상도 (s)
        """
        # 구성 요소 초기화
        self.scan_calculator = ScanTimeCalculator(
            min_doserate=min_doserate,
            max_speed=max_speed,
            min_speed=min_speed,
            time_resolution=time_resolution,
            doserate_table_path=DOSERATE_TABLE_PATH
        )
        
        self.treatment_calculator = TreatmentTimeCalculator(
            time_resolution=time_resolution
        )
        
        self.batch_processor = BatchProcessor(
            num_streams=Config.GPU_STREAMS,
            batch_size=Config.BATCH_SIZE
        )
        
        self.signal_generator = SignalGenerator(
            time_resolution=time_resolution
        )
        
        logger.info("하이브리드 구현 초기화 완료")
    
    def calculate_port_scan_times(self, port: Port, DR: Optional[float] = None) -> float:
        """
        포트의 모든 레이어에 대한 스캔 시간 계산 (위임 메서드)
        
        Args:
            port: 계산할 포트 객체
            DR: 선택적 선량율 조정값
            
        Returns:
            포트의 총 스캔 시간 (초)
        """
        return self.scan_calculator.calculate_port_scan_times(port, DR)

    def calculate_integrated_batch_with_data(self, 
                                          batch_data: BatchData, 
                                          DR: float) -> BatchResult:
        """
        BatchData 객체를 이용한 통합 계산 수행 (위임 메서드)
        
        Args:
            batch_data: 배치 데이터 객체
            DR: 선량율
            
        Returns:
            BatchResult: 계산 결과
        """
        return self.batch_processor.process_batch_data(
            batch_data=batch_data,
            treatment_calculator=self.treatment_calculator,
            DR=DR
        )
    
    def generate_gating_signal(self, gating_amplitude: float, resp_period: float) -> GatingPeriod:
        """게이팅 신호 생성 (위임 메서드)"""
        return self.signal_generator.generate_gating_signal(
            gating_amplitude=gating_amplitude,
            resp_period=resp_period
        )
    
    def calculate_all_dr_scan_times(self, port: Port, DR_list: List[float]) -> np.ndarray:
        """
        모든 DR 값에 대한 스캔 시간을 한 번에 계산
        
        Args:
            port: 계산할 포트 객체
            DR_list: DR 값 리스트
            
        Returns:
            [DR × 레이어] 차원의 스캔 시간 배열
        """
        return self.scan_calculator.calculate_all_dr_port_scan_times(port, DR_list)
    
    def calculate_integrated_full_batch(self, 
                                      port: Port,
                                      DR_list: List[float],
                                      TR_list: List[float], 
                                      amp_g_list: List[float],
                                      TLS_list: List[float]) -> Dict[Tuple[float, float, float, float], TreatmentTimeResult]:
        """
        전체 매개변수 조합에 대한 통합 계산 수행
        
        Args:
            port: 계산할 포트 객체
            DR_list: 선량율 리스트
            TR_list: 호흡 주기 리스트
            amp_g_list: 게이팅 진폭 리스트
            TLS_list: 레이어 전환 시간 리스트
            
        Returns:
            {(DR, TR, amp_g, TLS): TreatmentTimeResult} 형태의 결과 딕셔너리
        """
        logger.info(f"통합 배치 계산 시작 - 총 조합 수: {len(DR_list) * len(TR_list) * len(amp_g_list) * len(TLS_list)}")
        
        # 1. 모든 DR에 대한 스캔 시간 계산
        all_scan_times = self.calculate_all_dr_scan_times(port, DR_list)
        logger.info(f"스캔 시간 텐서 생성 완료: {all_scan_times.shape}")
        
        # 2. 게이팅 주기 정보 생성
        gating_periods = {}
        for TR in TR_list:
            for amp_g in amp_g_list:
                gating_period = self.signal_generator.generate_gating_signal(
                    gating_amplitude=amp_g,
                    resp_period=TR
                )
                gating_periods[(TR, amp_g)] = gating_period
        
        logger.info(f"게이팅 주기 정보 생성 완료: {len(gating_periods)}개")
        
        # 3. 전체 매개변수 조합 생성
        all_combinations = []
        for DR in DR_list:
            for TR in TR_list:
                for amp_g in amp_g_list:
                    for TLS in TLS_list:
                        all_combinations.append((DR, TR, amp_g, TLS))
        
        # 4. GPU 배치 처리로 전체 계산 수행
        results = self.batch_processor.process_full_parameter_space(
            scan_time_tensor=all_scan_times,
            DR_list=DR_list,
            parameter_combinations=all_combinations,
            gating_periods=gating_periods,
            treatment_calculator=self.treatment_calculator
        )
        
        logger.info(f"통합 배치 계산 완료: {len(results)}개 결과")
        return results

    def clear_caches(self) -> None:
        """캐시 메모리 정리 (위임 메서드)"""
        self.scan_calculator.clear_caches()