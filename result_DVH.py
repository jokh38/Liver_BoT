# result dvh 파일을 읽고 그래프를 그림
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class DVHParser:
    def __init__(self):
        """
        DVH 데이터 파서 초기화
        """
        self.files_data = {}  # {file_name: {roi_name: {dose, volume}}}
        self.all_roi_names = set()  # 전체 ROI 목록
        
    def parse_file(self, file_path):
        """
        단일 CSV 파일을 파싱하여 ROI별 DVH 데이터 추출
        
        Args:
            file_path (str or Path): 파싱할 파일 경로
        """
        file_path = Path(file_path)
        file_name = file_path.stem
        
        # 파일별 데이터 저장 공간 초기화
        self.files_data[file_name] = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='cp949') as file:
                content = file.read()
        
        # ROI별로 데이터 분할
        roi_sections = content.split('#RoiName:')[1:]  # 첫 번째 빈 요소 제거
        
        roi_count = 0
        for section in roi_sections:
            lines = section.strip().split('\n')
            if not lines:
                continue
                
            # ROI 이름 추출
            roi_name = lines[0].strip()
            self.all_roi_names.add(roi_name)  # 전체 ROI 목록에 추가
            
            # Unit 라인 찾기
            data_start_idx = -1
            for i, line in enumerate(lines):
                if line.startswith('#Unit:'):
                    data_start_idx = i + 1
                    break
            
            if data_start_idx == -1:
                continue
            
            # 데이터 추출
            dose_values = []
            volume_values = []
            
            for i in range(data_start_idx, len(lines)):
                line = lines[i].strip()
                if not line or line.startswith('#'):
                    break
                    
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        dose = float(parts[0])
                        volume = float(parts[1])
                        dose_values.append(dose)
                        volume_values.append(volume)
                except ValueError:
                    continue
            
            # ROI 데이터 저장
            if dose_values and volume_values:
                self.files_data[file_name][roi_name] = {
                    'dose': np.array(dose_values),
                    'volume': np.array(volume_values)
                }
                roi_count += 1
        
        print(f"파일 '{file_name}' 파싱 완료: {roi_count}개 ROI 발견")
        return list(self.files_data[file_name].keys())

    def get_all_roi_names(self):
        """
        모든 파일에서 발견된 ROI 이름 목록 반환
        
        Returns:
            list: 전체 ROI 이름 목록 (정렬됨)
        """
        return sorted(list(self.all_roi_names))
    
    def get_files_with_roi(self, roi_name):
        """
        특정 ROI를 포함하는 파일 목록 반환
        
        Args:
            roi_name (str): ROI 이름
            
        Returns:
            list: 해당 ROI를 포함하는 파일명 목록
        """
        files_with_roi = []
        for file_name in sorted(self.files_data.keys()):  # 정렬 추가
            if roi_name in self.files_data[file_name]:
                files_with_roi.append(file_name)
        return files_with_roi
    
    def plot_roi_comparison(self, selected_rois, figsize=(12, 8)):
        """
        선택된 ROI들을 파일별로 비교하여 그래프 생성
        
        Args:
            selected_rois (list): 그릴 ROI 목록
            figsize (tuple): 그래프 크기
        """
        if not selected_rois:
            print("선택된 ROI가 없습니다.")
            return
        
        plt.figure(figsize=figsize)
        
        # 색상과 라인 스타일 설정
        colors = plt.cm.Set2(np.linspace(0, 1, len(selected_rois)))
        line_styles = ['--', ':', '-']
        # 파일별 고정 스타일 매핑 생성
        all_files = sorted(self.files_data.keys())
        file_style_mapping = {file_name: line_styles[i % len(line_styles)] 
                             for i, file_name in enumerate(all_files)}
        
        legend_entries = []
        
        for roi_idx, roi_name in enumerate(selected_rois):
            files_with_roi = self.get_files_with_roi(roi_name)
            
            if not files_with_roi:
                print(f"'{roi_name}' ROI를 포함하는 파일이 없습니다.")
                continue
            
            for file_name in files_with_roi:
                data = self.files_data[file_name][roi_name]
                line_style = file_style_mapping[file_name] 
                
                plt.plot(data['dose'], data['volume'], 
                        color=colors[roi_idx], 
                        linestyle=line_style,
                        linewidth=2,
                        alpha=0.8)
                print(f'{file_name}_{roi_name}_dose_{data["dose"][80]}_volume_{data["volume"][80]}')
                legend_entries.append(f"{file_name}-{roi_name}")
        
        plt.xlabel('Dose (cGy)')
        plt.ylabel('Volume (%)')
        # plt.title('DVH Comparison - Multiple Files')
        plt.grid(True, alpha=0.3)
        plt.legend(legend_entries, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlim(left=0)
        plt.ylim(0, 105)
        
        # 레이아웃 조정
        plt.tight_layout()
        
        plt.show()

def main():
    """
    메인 실행 함수
    """
    # 디렉토리 경로 설정
    directory_path = r'C:\Users\com\OneDrive\2025 작업\Liver_BoT_DR\Final_code\data\DVH'
    directory = Path(directory_path)
    
    if not directory.exists():
        print("디렉토리가 존재하지 않습니다.")
        return
    
    # DVH 파서 생성
    parser = DVHParser()
    
    # 디렉토리 내 모든 .dvh 파일을 순차적으로 파싱
    dvh_files = sorted(list(directory.glob('*.dvh')))
    if not dvh_files:
        print("DVH 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(dvh_files)}개의 DVH 파일 발견")
    
    # 모든 파일 파싱
    for file_path in dvh_files:
        print(f"처리 중: {file_path.name}")
        parser.parse_file(file_path)
    
    # 파싱된 파일 요약 정보
    print(f"\n=== 파싱 완료 ===")
    print(f"총 파일 수: {len(parser.files_data)}")
    print(f"발견된 전체 ROI 수: {len(parser.all_roi_names)}")
    
    # 전체 ROI 목록 표시
    all_rois = parser.get_all_roi_names()
    print(f"\n=== 전체 ROI 목록 ===")
    for i, roi in enumerate(all_rois, 1):
        files_with_roi = parser.get_files_with_roi(roi)
        print(f"{i:2d}. {roi} (파일 수: {len(files_with_roi)})")
    
    # ROI별 파일 상세 정보 표시 (선택사항)
    # show_detail = input("\nROI별 파일 상세 정보를 보시겠습니까? (y/n): ").strip().lower()
    # if show_detail == 'y':
    #     print(f"\n=== ROI별 파일 상세 정보 ===")
    #     for roi in all_rois:
    #         files_with_roi = parser.get_files_with_roi(roi)
    #         print(f"{roi}: {', '.join(files_with_roi)}")
    
    # 사용자가 그릴 ROI 선택
    print(f"\n그래프에 포함할 ROI를 선택하세요:")
    print("- 전체: 'all' 입력")
    print("- 선택: 번호를 쉼표로 구분하여 입력 (예: 1,2,3)")
    print("- 이름: ROI 이름을 쉼표로 구분하여 입력")
    
    # selection = input("선택: ").strip()
    selection = [2,6,8,10,16,26,27,37]
    selected_rois = [all_rois[i-1] for i in selection]

    # if selection.lower() == 'all':
    #     selected_rois = all_rois
    # elif selection.replace(',', '').replace(' ', '').isdigit():
    #     # 번호로 선택
    #     indices = [int(x.strip()) - 1 for x in selection.split(',')]
    #     selected_rois = [all_rois[i] for i in indices if 0 <= i < len(all_rois)]
    # else:
    #     # 이름으로 선택
    #     roi_names = [x.strip() for x in selection.split(',')]
    #     selected_rois = [roi for roi in roi_names if roi in all_rois]
    
    # if not selected_rois:
    #     print("선택된 ROI가 없습니다.")
    #     return
    
    print(f"\n선택된 ROI: {selected_rois}")
    
    # 선택된 ROI별 파일 정보 표시
    # print(f"\n=== 선택된 ROI별 파일 정보 ===")
    # for roi in selected_rois:
    #     files_with_roi = parser.get_files_with_roi(roi)
    #     print(f"{roi}: {files_with_roi}")
    
    # 그래프 생성
    parser.plot_roi_comparison(selected_rois)

if __name__ == "__main__":
    main()