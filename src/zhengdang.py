import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf



# 示例时序数据，可以替换为您的实际数据
#data = np.sin(np.linspace(0, 20 * np.pi, 100))
data = [1,3,1,3,1,3,1,3,1,3,1,3,1,3]
# 计算自相关函数
acf_values = acf(data)

# 绘制自相关函数图像
plt.plot(acf_values)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function')
plt.show()

# 计算振荡强度指标，这里取前10个滞后作为主要周期
oscillation_strength = np.mean(acf_values[1:11])  # 排除滞后0
print("Oscillation Strength:", oscillation_strength)
