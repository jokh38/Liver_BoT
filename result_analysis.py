# result csv 파일을 읽고 그래프를 그림
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from tqdm import tqdm
from scipy.interpolate import interp1d, PchipInterpolator
import seaborn as sns

# 폴더 내 모든 csv 파일을 읽기
def find_csv_in_folder(folder):
    # 폴더 내 모든 csv 파일을 읽기
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    if len(csv_files) == 0:
        raise FileNotFoundError("No csv file found in the folder")
    else:
        return csv_files

def read_csv_in_folder(folder):
    csv_files = find_csv_in_folder(folder)
    result_csv = pd.DataFrame()
    for file in csv_files:
        result_csv = pd.concat([result_csv, pd.read_csv(file)])
    return result_csv

def sort_with_progress(df, columns):
    """데이터를 청크 단위로 정렬하여 진행 상황 표시"""
    chunk_size = 10000  # 청크 크기 설정
    chunks = []
    
    # 데이터를 청크 단위로 나누기
    for i in tqdm(range(0, len(df), chunk_size), desc="Sorting chunks"):
        chunk = df.iloc[i:i + chunk_size]
        chunks.append(chunk.sort_values(columns))
    
    # 모든 청크를 합치기
    sorted_df = pd.concat(chunks)
    return sorted_df.sort_values(columns)  # 마지막으로 전체 정렬

def plot_efficiency_analysis(result_csv, conditions_dict, x_column, y_column, plot_title, ax=None, enable_interpolation=True, y_limits=None):
    """
    통합된 효율성 분석 그래프 생성 함수
    
    Parameters:
    - result_csv: 결과 데이터프레임
    - conditions_dict: 필터링 조건 딕셔너리 (예: {'layer_switching_time': 0, 'gating_amplitude': 0.2})
    - x_column: X축 컬럼명
    - y_column: Y축 컬럼명
    - plot_title: 그래프 제목
    - ax: matplotlib axes 객체
    - enable_interpolation: 보간 사용 여부
    - y_limits: Y축 범위 (tuple)
    """
    # 조건에 따라 데이터 필터링
    filtered_data = result_csv.copy()
    for column, value in conditions_dict.items():
        if column in filtered_data.columns:
            filtered_data = filtered_data[filtered_data[column] == value]
    
    if len(filtered_data) == 0:
        conditions_str = ', '.join([f"{k}={v}" for k, v in conditions_dict.items()])
        print(f"No data found for conditions: {conditions_str}")
        return
    
    # 데이터 정렬
    sort_columns = ['filename', 'port_name']
    filtered_data = sort_with_progress(filtered_data, sort_columns)
    
    # 그룹화
    grouped = filtered_data.groupby(['filename', 'port_name'])
    
    if ax is None:
        ax = plt.gca()
    
    color_index = 0
    
    # 각 그룹별 그래프 생성
    for (filename, port_name), group in grouped:
        group_sorted = group.sort_values(x_column)
        x_data = group_sorted[x_column].values
        y_data = group_sorted[y_column].values
        
        if len(x_data) == 0:
            continue

        if color_index % 2 == 0:    
            plot_style_1 = '-'
        else:
            plot_style_1 = '--'
        
        color = f'C{color_index}'
        
        if enable_interpolation and len(x_data) >= 2:
            # 보간 수행
            x_min, x_max = x_data.min(), x_data.max()
            x_step = (x_max - x_min) / 100 if x_max != x_min else 0.1
            x_interp = np.arange(x_min, x_max + x_step, max(x_step, 0.1))
            
            try:
                interp_func = interp1d(x_data, y_data, kind='linear', 
                                     bounds_error=False, fill_value='extrapolate')
                y_interp = interp_func(x_interp)
                
                # 보간된 데이터 플롯
                ax.plot(x_interp, y_interp, plot_style_1, linewidth=1, 
                       label=f'{filename} {port_name}', color=color)
                # 원본 데이터 점 플롯
                # ax.plot(x_data, y_data, 'o', markersize=2, 
                #        label=f'{filename} {port_name}', color=color)
                ax.plot(x_data, y_data, 'o', markersize=2, color=color)
            except Exception as e:
                # 보간 실패 시 원본 데이터만 플롯
                ax.plot(x_data, y_data, 'o-', linewidth=1, markersize=2, 
                       label=f'{filename} {port_name}', color=color)
        else:
            # 보간 없이 원본 데이터만 플롯
            ax.plot(x_data, y_data, 'o-', linewidth=1, markersize=2, 
                   label=f'{filename} {port_name}', color=color)
        
        color_index += 1
    
    # 축 라벨 및 제목 설정
    if x_column == 'doserate':
        ax.set_xlabel('Dynamic range')
    else    :
        ax.set_xlabel(x_column.replace('_', ' ').title())
    
    if y_column == 'efficiency':
        ax.set_ylabel('Efficiency')
    else:
        ax.set_ylabel(y_column.replace('_', ' ').title())
    
    ax.set_title(plot_title)
    ax.grid(True)

    if y_limits:
        ax.set_ylim(y_limits)

def plot_beam_on_time_by_dr(result_csv, conditions_dict=None, layer_switching_threshold=0, 
                           plot_type='line', save_plot=False, output_path='beam_time_plot.png'):
    """
    동일한 filename, port_name 쌍에 대해 doserate 변화에 따른 beam_on_time 그래프 생성
    
    Parameters:
    - conditions_dict: 추가 필터링 조건 (dict)
    - layer_switching_threshold: layer_switching_time 임계값 (기본값: 0)
    - plot_type: 그래프 타입 ('line', 'scatter', 'box', 'violin')
    - save_plot: 그래프 저장 여부
    - output_path: 저장 경로
    """
    
    # 기본 조건 적용
    filtered_data = result_csv.copy()
    
    # 추가 조건 필터링
    if conditions_dict:
        for column, value in conditions_dict.items():
            if column in filtered_data.columns:
                initial_count = len(filtered_data)
                filtered_data = filtered_data[filtered_data[column] == value]
                print(f"Filter {column}={value}: {initial_count} -> {len(filtered_data)} records")
    
    # layer_switching_time 조건 적용
    initial_count = len(filtered_data)
    filtered_data = filtered_data[filtered_data['layer_switching_time'] == layer_switching_threshold]
    print(f"Filter layer_switching_time={layer_switching_threshold}: {initial_count} -> {len(filtered_data)} records")
    
    if len(filtered_data) == 0:
        print("No data found matching the conditions")
        return
    
    # 데이터 정렬
    print("Sorting data...")
    sort_columns = ['filename', 'port_name', 'doserate']
    available_columns = [col for col in sort_columns if col in filtered_data.columns]
    filtered_data = sort_with_progress(filtered_data, available_columns)
    
    # filename, port_name 쌍별로 그룹화
    print("Creating plots...")
    file_port_groups = filtered_data.groupby(['filename', 'port_name'])
    
    # 그래프 설정
    plt.figure(figsize=(13, 9))
    plt.style.use('default')
    
    for idx, ((filename, port_name), group) in enumerate(file_port_groups, 1):
        
        # doserate별 정렬
        group_sorted = group.sort_values('doserate')
        doserates = group_sorted['doserate'].values
        beam_times = group_sorted['beam_on_time'].values
        
        if plot_type == 'line':
            plt.plot(doserates, beam_times, 'o--', linewidth=0.5, markersize=2)
        elif plot_type == 'scatter':
            plt.scatter(doserates, beam_times, s=30, alpha=0.7)
        elif plot_type == 'box':
            # doserate별 그룹화하여 box plot
            dr_groups = group.groupby('doserate')['beam_on_time'].apply(list).to_dict()
            plt.boxplot(list(dr_groups.values()), labels=list(dr_groups.keys()))
        elif plot_type == 'violin':
            # violin plot을 위한 데이터 준비
            sns.violinplot(data=group, x='doserate', y='beam_on_time')

        # label = custom_labels.get((filename, port_name), f'{filename} {port_name}')
        # ax.plot(..., label=label)
        
    plt.xlabel('Dynamic range')
    plt.ylabel('Beam On Time')
    # plt.title('Beam On Time by Dynamic range')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    # plt.show()

# 사용 예시
if __name__ == "__main__":
    result_csv = read_csv_in_folder("./results/DR_pat")

    # subplot 레이아웃 설정
    dr_values = [10, 50, 90, 110, 130, 170]  # 5개 DR 값
    # dr_values = list(range(10, 230, 20))  # 11개 DR 값
    tr_values = [1.01, 2.01, 3.01, 4.01, 5.01, 6.01]  # 6개 TR 값

    # 출력 디렉토리 생성
    output_dir = "./output_plots"
    os.makedirs(output_dir, exist_ok=True)

# --------------------------------------------

    # TR별 doserate에 따른 efficiency plot (2x3 레이아웃)
    fig1, axes1 = plt.subplots(3, 2, figsize=(12, 18))
    # fig1.suptitle('Efficiency by Doserate for Different TR values')

    Tls = 2
    amp_g = 0.2
    
    for i, tr_val in enumerate(tr_values):
        row, col = i // 2, i % 2
        conditions = {
            'layer_switching_time': Tls,
            'gating_amplitude': amp_g,
            'resp_period': tr_val
        }
        plot_title = f'TR={tr_val}'
        plot_efficiency_analysis(result_csv, conditions, 'doserate', 'total_time', 
                               plot_title, ax=axes1[row, col], enable_interpolation=True)

        # Add legend to each subplot
        axes1[row, col].legend(loc='best', fontsize='x-small')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'efficiency_by_DR_plots_LST{Tls}_amp{amp_g}.jpg'), dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # DR별 resp_period에 따른 efficiency plot (2x3 레이아웃)
    fig2, axes2 = plt.subplots(3, 2, figsize=(12, 18))
    # fig2.suptitle('Efficiency by Response Period for Different dynamic range values')

    for i, dr_val in enumerate(dr_values[:9]):  # 9개 DR 값만 처리
        row, col = i // 2, i % 2
        conditions = {
            'layer_switching_time': Tls,
            'gating_amplitude': amp_g,
            'doserate': dr_val
        }
        plot_title = f'Dynamic range={dr_val}'
        # plot_efficiency_analysis(result_csv, conditions, 'resp_period', 'efficiency', 
        #                        plot_title, ax=axes2[row, col], enable_interpolation=True,
        #                        y_limits=(0.49, 1.01))
        plot_efficiency_analysis(result_csv, conditions, 'resp_period', 'total_time', 
                                plot_title, ax=axes2[row, col], enable_interpolation=True)

        # Add legend to each subplot
        axes2[row, col].legend(loc='best', fontsize='x-small')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'efficiency_by_TR_plots_LST{Tls}_amp{amp_g}.jpg'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    # -------------------------------------------
    

    # 라인 그래프로 표시
    plot_beam_on_time_by_dr(
        result_csv,
        plot_type='line',
        save_plot=True
    )

    print("Plots saved successfully!")
