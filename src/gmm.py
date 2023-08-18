import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# 生成示例时序数据
np.random.seed(42)
num_time_steps = 100
time_series_data = np.random.randn(num_time_steps, 1)  # 一维时序数据

# 进行慢特征分析，提取慢特征信息和慢特征差分信息
def extract_slow_features(data):
    pca = PCA(n_components=1)  # 降维到1维
    sfa_features = pca.fit_transform(data)
    return sfa_features

# 使用高斯混合模型拟合特征数据的分布
def fit_gmm(features, num_components):
    gmm = GaussianMixture(n_components=num_components)
    gmm.fit(features)
    return gmm

# 提取基准训练数据的慢特征信息和慢特征差分信息
baseline_sfa_features = extract_slow_features(time_series_data)

# 使用高斯混合模型拟合基准训练数据的分布
num_components = 2
gmm = fit_gmm(baseline_sfa_features, num_components)

# 生成测试时序数据
test_time_series_data = np.random.randn(num_time_steps, 1)  # 一维时序数据

# 提取测试数据的慢特征信息和慢特征差分信息
test_sfa_features = extract_slow_features(test_time_series_data)

# 计算Hellinger距离
def calculate_hellinger_distance(pdf1, pdf2):
    m = 0.5 * (pdf1 + pdf2)
    distance = np.sqrt(1.0 - np.sqrt(np.minimum(pdf1, pdf2) / m))
    return distance

# 计算测试数据的概率密度函数
baseline_pdf = np.exp(gmm.score_samples(baseline_sfa_features))
test_pdf = np.exp(gmm.score_samples(test_sfa_features))

# 计算Hellinger距离
distance = calculate_hellinger_distance(baseline_pdf, test_pdf)

# 绘制基准数据与测试数据的时序图
plt.plot(time_series_data, label="Baseline Data")
plt.plot(test_time_series_data, label="Test Data")
plt.legend()
plt.title("Time Series Data")
plt.xlabel("Time Step")
plt.ylabel("Coal Temperature")
plt.show()
max_possible_distance = np.sqrt(2)  # 最大可能距离
normalized_distance = distance / max_possible_distance
print("Hellinger距离：", normalized_distance.item())
