"""
GPU 메모리 관리와 연산을 위한 유틸리티 클래스
"""
import cupy as cp
import logging
from typing import Any, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

class GPUMemoryManager:
    """GPU 메모리 관리를 위한 유틸리티 클래스"""
    
    def __init__(self):
        """GPU 메모리 관리자 초기화"""
        try:
            self.device = cp.cuda.Device(0)
            self.device.use()
            self.memory_pool = cp.cuda.MemoryPool()
            cp.cuda.set_allocator(self.memory_pool.malloc)
            
            # GPU 정보 로깅
            gpu_info = cp.cuda.runtime.getDeviceProperties(0)
            logger.info(f"GPU 초기화: {gpu_info['name'].decode()}")
            logger.info(f"CUDA 코어: {gpu_info['multiProcessorCount'] * 128}")
            logger.info(f"가용 메모리: {self.device.mem_info[0]/1024**3:.2f}GB")
        except (ImportError, AttributeError, RuntimeError) as e:
            error_msg = f"GPU 초기화 실패: {e}. 이 모듈은 GPU가 필요합니다."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def clean_memory(self):
        """GPU 메모리 정리"""
        try:
            self.memory_pool.free_all_blocks()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            cp.cuda.runtime.deviceSynchronize()
            logger.debug("GPU 메모리 정리 완료")
        except Exception as e:
            logger.warning(f"GPU 메모리 정리 중 오류: {e}")
    
    @staticmethod
    def array_to_gpu(array: np.ndarray, dtype=cp.float32) -> Optional[cp.ndarray]:
        """NumPy 배열을 GPU로 전송"""
        try:
            return cp.array(array, dtype=dtype)
        except Exception as e:
            logger.error(f"GPU 배열 변환 실패: {e}")
            return None
            
    @staticmethod
    def array_to_cpu(gpu_array: cp.ndarray) -> Optional[np.ndarray]:
        """GPU 배열을 CPU로 전송"""
        try:
            return cp.asnumpy(gpu_array)
        except Exception as e:
            logger.error(f"CPU 배열 변환 실패: {e}")
            return None
    
    def get_memory_info(self) -> tuple:
        """현재 GPU 메모리 사용 정보 반환"""
        try:
            free_memory, total_memory = self.device.mem_info
            return free_memory, total_memory
        except Exception as e:
            logger.error(f"GPU 메모리 정보 획득 실패: {e}")
            return 0, 0