# 在cmd输入以下指令一键安装依赖库：
# pip install pandas numpy scipy tqdm

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from tqdm import tqdm

# 读取数据
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 构建索引
user_list = train['user_id'].unique()
item_list = train['item_id'].unique()
user_to_idx = {u: i for i, u in enumerate(user_list)}
item_to_idx = {i: j for j, i in enumerate(item_list)}
idx_to_item = {j: i for i, j in item_to_idx.items()}

train['user_idx'] = train['user_id'].map(user_to_idx)
train['item_idx'] = train['item_id'].map(item_to_idx)

# 稀疏矩阵
n_users = len(user_to_idx)
n_items = len(item_to_idx)
matrix = coo_matrix((np.ones(len(train)), (train['user_idx'], train['item_idx'])),
                    shape=(n_users, n_items)).tocsr()

# SVD 分解
k = 20
user_factors, sigma, item_factors = svds(matrix, k=k)
user_factors = user_factors @ np.diag(sigma)

# 用户历史
user_history = train.groupby('user_id')['item_id'].apply(set).to_dict()

# 图书流行度（用于 rerank）
item_popularity = train['item_id'].value_counts(normalize=True).to_dict()

# 推荐函数：召回 top_k 个候选，再 rerank
def recommend_rerank(user_ids, top_k=20):
    recommendations = []
    for user in tqdm(user_ids, desc="推荐中"):
        if user not in user_to_idx:
            # 冷启动：返回最热门 item
            top_item = train['item_id'].value_counts().idxmax()
            recommendations.append(top_item)
            continue
        
        u_idx = user_to_idx[user]
        scores = item_factors.T @ user_factors[u_idx]

        seen_items = user_history.get(user, set())
        for item in seen_items:
            if item in item_to_idx:
                scores[item_to_idx[item]] = -np.inf

        # top_k 个候选
        top_k_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_k_items = [(idx_to_item[i], scores[i]) for i in top_k_idx]

        # rerank：按流行度加权重新排序（你可以改为其它策略）
        reranked = sorted(top_k_items, key=lambda x: x[1] + 0.5 * item_popularity.get(x[0], 0), reverse=True)

        final_item = reranked[0][0]
        recommendations.append(final_item)
    
    return recommendations

# 生成推荐
test_user_ids = test['user_id'].tolist()
recommended_items = recommend_rerank(test_user_ids, top_k=20)

# 保存结果
submission = pd.DataFrame({
    'user_id': test_user_ids,
    'item_id': recommended_items
})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv 已生成，使用多物品召回+rerank 完成推荐。")
