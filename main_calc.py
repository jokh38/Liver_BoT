"""
DICOM 파일 처리 및 결과 저장 메인 모듈
"""
import os
import logging
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

# 로컬 모듈 임포트
from config import Config, ROOT_DIR, RESULTS_DIR
from core.dicom_loader import DicomLoader
from core.hybrid_implementation import HybridImplementation
from data_models.calculation_data import BatchData, GatingPeriod

# 로깅 설정
logger = logging.getLogger(__name__)

def process_treatment_data(file_path):
    """
    DICOM 파일 로드부터 계산 및 결과 저장까지 수행
    
    Args:
        file_path (str): DICOM 파일 경로
        
    Returns:
        int: 생성된 결과 개수
    """
    try:
        # 파일명 추출
        filename = os.path.basename(file_path)
        logger.info(f"처리 시작: {filename}")
        
        # DICOM 파일 로드
        dicom_loader = DicomLoader(file_path)
        ports = dicom_loader.get_ports()
        
        if not ports:
            logger.error(f"포트 정보가 없습니다: {filename}")
            return 0
            
        # Config에서 매개변수 가져오기
        DR_list = Config.DR
        TR_list = Config.TR
        amp_g_list = Config.amp_g
        TLS_list = Config.TLS
        
        logger.info(f"계산 매개변수: DR={DR_list}, TR={TR_list}, amp_g={amp_g_list}, TLS={TLS_list}")
        
        # HybridImplementation 객체 생성
        hybrid_impl = HybridImplementation(
            min_doserate=Config.MIN_DOSERATE,
            max_speed=Config.MAX_SPEED,
            min_speed=Config.MIN_SPEED,
            time_resolution=Config.TIME_RESOLUTION
        )
        
        # 각 포트별로 계산 수행 (GPU 통합 방식)
        all_results = []
        
        for i_port, port in enumerate(ports):            
            try:
                port_name = port.BeamName
            except:
                port_name = f"Port_{i_port+1}"
            
            logger.info(f"포트 처리 중: {port_name} (레이어 수: {len(port.layers)})")
            
            # GPU 통합 계산 수행
            logger.info(f"GPU 통합 계산 시작 - DR: {len(DR_list)}개, TR: {len(TR_list)}개, amp_g: {len(amp_g_list)}개, TLS: {len(TLS_list)}개")
            
            # 전체 매개변수 조합에 대한 통합 계산
            full_batch_result = hybrid_impl.calculate_integrated_full_batch(
                port=port,
                DR_list=DR_list,
                TR_list=TR_list,
                amp_g_list=amp_g_list,
                TLS_list=TLS_list
            )

            total_mu = sum([sum(layer.weights) for layer in port.get_layers()])
            
            # 결과 추출 및 저장
            for (DR, TR, amp_g, TLS), treatment_result in full_batch_result.items():
                # 결과 딕셔너리 생성
                result_dict = {
                    'filename': filename,
                    'port_name': port_name,
                    'total_layers': len(port.layers),
                    'total_MU': total_mu,
                    'doserate': DR,
                    'resp_period': TR,
                    'gating_amplitude': amp_g,
                    'layer_switching_time': TLS,
                    'total_time': treatment_result.total_time,
                    'beam_on_time': treatment_result.beam_on_time,
                    'efficiency': treatment_result.efficiency
                }
                
                all_results.append(result_dict)
            
            logger.info(f"포트 {port_name} 처리 완료: {len(full_batch_result)}개 결과 생성")
        
        # 데이터프레임 생성 및 CSV 저장
        if all_results:
            results_df = pd.DataFrame(all_results)
            
            # 저장 경로 설정
            output_filename = f"{os.path.splitext(filename)[0]}_results.csv"
            output_path = os.path.join(RESULTS_DIR, output_filename)
            
            # CSV 파일로 저장
            results_df.to_csv(output_path, index=False)
            logger.info(f"결과 저장 완료: {output_path} ({len(all_results)}개 결과)")
            
            # 메모리 정리
            del results_df
            hybrid_impl.clear_caches()
            
            return len(all_results)
        else:
            logger.warning(f"결과가 없습니다: {filename}")
            return 0
            
    except Exception as e:
        logger.exception(f"처리 중 오류 발생: {file_path}")
        return 0

def main():
    """프로그램 진입점"""
    # 디렉토리 확인
    Config.ensure_directories()
    
    # 명령행 인자 처리
    parser = argparse.ArgumentParser(description='DICOM 파일 처리 및 계산')
    parser.add_argument('--file', type=str, help='처리할 DICOM 파일 경로')
    parser.add_argument('--dir', type=str, help='처리할 DICOM 파일이 있는 디렉토리 경로')
    args = parser.parse_args()
    
    # 콘솔 로그 설정
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    
    if args.file:
        # 단일 파일 처리
        if not os.path.exists(args.file):
            logger.error(f"파일을 찾을 수 없습니다: {args.file}")
            return
            
        count = process_treatment_data(args.file)
        logger.info(f"처리 완료: {count}개 결과 생성")
        
    elif args.dir:
        # 디렉토리 내 모든 DICOM 파일 처리
        if not os.path.exists(args.dir):
            logger.error(f"디렉토리를 찾을 수 없습니다: {args.dir}")
            return
            
        # .dcm 확장자 파일 찾기
        dicom_files = []
        for root, _, files in os.walk(args.dir):
            for file in files:
                if file.lower().endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))
        
        if not dicom_files:
            logger.error(f"디렉토리에 DICOM 파일이 없습니다: {args.dir}")
            return
            
        total_count = 0
        for file_path in tqdm(dicom_files, desc="DICOM 파일 처리 중"):
            count = process_treatment_data(file_path)
            total_count += count
            
        logger.info(f"전체 처리 완료: {len(dicom_files)}개 파일, {total_count}개 결과 생성")
        
    else:
        parser.print_help()
        logger.error("--file 또는 --dir 인자가 필요합니다.")

if __name__ == "__main__":
    import sys
    # data 폴더 경로 설정
    DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    if len(sys.argv) == 1:  # 인자가 없는 경우
        sys.argv.extend(["--dir", DEFAULT_DIR])
        print(f"기본 폴더 사용: {DEFAULT_DIR}")
    
    main()