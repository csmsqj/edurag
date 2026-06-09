from sentence_transformers import CrossEncoder
import os
absfile=os.path.abspath(__file__)
rootfile=os.path.dirname(os.path.dirname(absfile))
file=os.path.join(rootfile,'models/bge-reranker-large')
# 加载重排序模型（用相对路径）
model = CrossEncoder(model_name_or_path=file)

# 构造 query-doc 对
pairs = [
    ['what is panda?', 'what is panda?'],      # 完全匹配
    ['what is panda?', 'hi'],                   # 完全不相关
    ['what is panda?', 'The giant panda '],     # 相关
]

# 预测相关性分数
print(model.predict(pairs))