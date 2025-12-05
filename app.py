from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import koreanize_matplotlib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def read_csv_file(file_storage):
    try:
        return pd.read_csv(file_storage, encoding='utf-8')
    except UnicodeDecodeError:
        file_storage.seek(0)
        try:
            return pd.read_csv(file_storage, encoding='cp949')
        except UnicodeDecodeError:
            file_storage.seek(0)
            return pd.read_csv(file_storage, encoding='euc-kr')

@app.route('/preview', methods=['POST'])
def preview():
    try:
        file = request.files['file']
        df = read_csv_file(file)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        preview_data = df.head().fillna('').to_dict(orient='records')
        return jsonify({
            'success': True,
            'columns': numeric_cols,
            'preview_data': preview_data,
            'total_rows': len(df)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # 1. 파라미터 받기
        file = request.files['file']
        k = int(request.form.get('k', 3))
        x_axis = request.form.get('x_axis')
        y_axis = request.form.get('y_axis')
        use_pca = request.form.get('use_pca') == 'true'
        normalize = request.form.get('normalize') == 'true'

        # 2. 데이터 준비
        df = read_csv_file(file)
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        
        if x_axis not in numeric_df.columns or y_axis not in numeric_df.columns:
            return jsonify({'success': False, 'error': '선택한 컬럼 없음'})

        X = numeric_df.values
        
        # 3. 데이터 변환 (계산용 및 시각화용)
        # ----------------------------------------------------
        scaler = StandardScaler()
        
        # (1) 정규화 적용 여부
        if normalize:
            # 정규화 체크 시: 데이터 전체를 변환 (평균=0, 분산=1)
            X_transformed = scaler.fit_transform(X)
            
            # 시각화할 데이터도 변환된 값에서 가져옴
            x_col_idx = numeric_df.columns.get_loc(x_axis)
            y_col_idx = numeric_df.columns.get_loc(y_axis)
            
            x_data = X_transformed[:, x_col_idx]
            y_data = X_transformed[:, y_col_idx]
            
            axis_suffix = " (Standardized)"
        else:
            # 정규화 미체크 시: 원본 데이터 사용
            X_transformed = X
            x_data = numeric_df[x_axis]
            y_data = numeric_df[y_axis]
            axis_suffix = " (Original)"

        # (2) PCA 적용 여부 (군집화 계산에만 영향, 시각화 축은 사용자가 선택한 X,Y 유지)
        if use_pca:
            # PCA는 이미 정규화된(혹은 원본) X_transformed를 입력으로 받음
            pca_calc = PCA(n_components=2)
            X_calc = pca_calc.fit_transform(X_transformed)
        else:
            X_calc = X_transformed

        # 4. K-means 수행
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_calc)
        
        # 5. 시각화
        plt.figure(figsize=(10, 8))
        
        scatter = plt.scatter(x_data, y_data, c=labels, cmap='viridis', 
                              alpha=0.7, edgecolors='w', s=60)
        
        # 중심점 표시 로직
        if normalize:
            # 정규화된 상태면, 중심점도 정규화된 좌표계(kmeans.cluster_centers_)에서 가져와야 함
            # 단, PCA까지 썼다면 역변환이 복잡하므로, 가장 확실한 방법인 '그룹별 평균'을 사용
            temp_df = pd.DataFrame({ 'x': x_data, 'y': y_data, 'label': labels })
            centers = temp_df.groupby('label').mean().values
        else:
            # 원본이면 원본 데이터의 그룹별 평균
            temp_df = pd.DataFrame({ 'x': x_data, 'y': y_data, 'label': labels })
            centers = temp_df.groupby('label').mean().values

        plt.scatter(centers[:, 0], centers[:, 1], 
                   c='red', marker='X', s=200, 
                   edgecolors='black', linewidths=2, 
                   label='Cluster Center')
        
        plt.xlabel(x_axis + axis_suffix, fontsize=12, fontweight='bold')
        plt.ylabel(y_axis + axis_suffix, fontsize=12, fontweight='bold')
        plt.title(f'K-means Result (K={k})', fontsize=14)
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 6. 결과 반환
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        numeric_df['Cluster'] = labels
        cluster_counts = numeric_df['Cluster'].value_counts().sort_index().to_dict()
        
        return jsonify({
            'success': True,
            'plot_url': plot_url,
            'cluster_counts': cluster_counts,
            'inertia': float(kmeans.inertia_)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)