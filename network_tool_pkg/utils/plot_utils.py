import numpy as np
import matplotlib.pyplot as plt

# -------------------- Degree 비교 플롯을 위한 함수 --------------------
def plot_degree_hist(ax, original, model, model_name, bar_alpha = 0.35, original_color = 'blue', model_color = 'red') :

  k = np.arange(len(original))

  # ---------- 원본 히스토그램 생성 ----------

  bins = range(max(original) + 2)
  original_hist = np.histogram(original, bins = bins, density = True)[0]

  # ---------- 원본 bar + line 생성 ----------

  ax.bar(k - 0.2, original_hist, width = 0.4, alpha = bar_alpha, color = original_color, label = 'Original (bar)')
  ax.plot(k - 0.2, original_hist, color = original_color, linewidth = 2, label = 'Original (line)')

  # ---------- 랜덤 모델 bar + line 생성 ----------

  ax.bar(k + 0.2, model, width = 0.4, alpha = bar_alpha, color = model_color, label = '{} (bar)'.format(model_name))
  ax.plot(k + 0.2, model, color = model_color, linestyle = '--', linewidth = 2, label = '{} (line)'.format(model_name))

  # ---------- 데코레이션 ----------

  ax.set_xlabel(r'Degree $k$')
  ax.set_ylabel(r'$P(k)$')
  ax.set_title('Original vs {}'.format(model_name))
  ax.legend()
  ax.grid(alpha = 0.4)
