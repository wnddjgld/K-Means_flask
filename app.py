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

        # 2. 데이터 준비
        df = read_csv_file(file)
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        
        if x_axis not in numeric_df.columns or y_axis not in numeric_df.columns:
            return jsonify({'success': False, 'error': '선택한 컬럼 없음'})

        X = numeric_df.values
        
        # 3. 정규화 (항상 적용)
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)

        # 4. K-means 수행 (정규화된 데이터로)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_normalized)
        
        # 5. 시각화 준비
        # 원본 데이터의 X, Y 추출
        x_data_original = numeric_df[x_axis].values
        y_data_original = numeric_df[y_axis].values
        
        # PCA 변환
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_normalized)
        
        # 6. 그래프 1: 원본 X, Y로 산점도
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 색상 매핑 (viridis 컬러맵)
        from matplotlib.cm import viridis
        colors = viridis(np.linspace(0, 1, k))
        
        # 그래프 1 - 각 군집별로 색상을 지정하여 범례에 표시
        for cluster_id in range(k):
            mask = labels == cluster_id
            ax1.scatter(x_data_original[mask], y_data_original[mask], 
                       c=[colors[cluster_id]], alpha=0.7, edgecolors='w', s=60,
                       label=f'Cluster {cluster_id}')
        
        # 원본 데이터의 군집별 중심점
        temp_df1 = pd.DataFrame({'x': x_data_original, 'y': y_data_original, 'label': labels})
        centers1 = temp_df1.groupby('label')[['x', 'y']].mean().values
        
        ax1.scatter(centers1[:, 0], centers1[:, 1], 
                   c='red', marker='X', s=200, 
                   edgecolors='black', linewidths=2, 
                   label='중심점 (Centroid)')
        
        ax1.set_xlabel(x_axis + ' (Original)', fontsize=12, fontweight='bold')
        ax1.set_ylabel(y_axis + ' (Original)', fontsize=12, fontweight='bold')
        ax1.set_title(f'K-means Result - Original Data (K={k})', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=9, framealpha=0.9)
        
        # 7. 그래프 2: PCA 2D 산점도
        for cluster_id in range(k):
            mask = labels == cluster_id
            ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[colors[cluster_id]], alpha=0.7, edgecolors='w', s=60,
                       label=f'Cluster {cluster_id}')
        
        # PCA 공간의 군집별 중심점
        temp_df2 = pd.DataFrame({'pc1': X_pca[:, 0], 'pc2': X_pca[:, 1], 'label': labels})
        centers2 = temp_df2.groupby('label')[['pc1', 'pc2']].mean().values
        
        ax2.scatter(centers2[:, 0], centers2[:, 1], 
                   c='red', marker='X', s=200, 
                   edgecolors='black', linewidths=2, 
                   label='중심점 (Centroid)')
        
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                      fontsize=12, fontweight='bold')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                      fontsize=12, fontweight='bold')
        ax2.set_title(f'K-means Result - PCA 2D (K={k})', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=9, framealpha=0.9)
        
        plt.tight_layout()

        # 8. 결과 반환
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        numeric_df['Cluster'] = labels
        cluster_counts = numeric_df['Cluster'].value_counts().sort_index().to_dict()
        
        return jsonify({
            'success': True,
            'plot_url': plot_url,
            'cluster_counts': cluster_counts,
            'inertia': float(kmeans.inertia_),
            'pca_variance': [float(pca.explained_variance_ratio_[0]), 
                           float(pca.explained_variance_ratio_[1])]
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)