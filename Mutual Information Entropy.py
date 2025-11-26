import numpy as np
import math
import itertools
import pandas as pd

def calculate_entropy_log10(probabilities):
    """
    计算香农熵 (单位: Hartleys, base 10)
    公式: E = - sum(P * log10(P))
    """
    entropy = 0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log10(p)
    return entropy

def calculate_custom_t_value(target_vars, df_details):
    """
    用算法计算 T 值 (基于不相交区域的 entropy 值交替求和)
    公式: Sum(En of 1-var subsets) - Sum(En of 2-var subsets) + ... 
    
    参数:
    target_vars: 目标变量组合，例如 ['A', 'I']
    df_details: 包含 'Region' 和 'En' 列的 DataFrame
    """
    target_set = set(target_vars)
    t_val = 0
    
    # 遍历所有不相交区域
    for _, row in df_details.iterrows():
        region_key = row['Region']
        en_value = row['Entropy']
        
        # 解析该区域包含的变量 (例如 'IA' -> {'I', 'A'})
        region_vars = set(list(region_key))
        
        # 检查该区域是否完全由目标变量组成 (即 region 是 target 的子集)
        if region_vars.issubset(target_set):
            k = len(region_vars)
            # 符号逻辑: 
            # 1个变量 (k=1) -> +
            # 2个变量 (k=2) -> -
            # 3个变量 (k=3) -> +
            # 4个变量 (k=4) -> -
            # 通式: (-1)^(k-1)
            sign = (-1) ** (k - 1)
            t_val += sign * en_value
            
    return t_val

def run_optimization_analysis(data_dict):
    """
    执行完整分析，返回两个 DataFrame
    """
    total = sum(data_dict.values())
    
    # ================= 1. 构建详细熵值表 (df_details) =================
    # 定义数据的严格顺序，对应切片逻辑
    order_keys = [
        'A', 'I', 'S', 'G',                  # 0-3 (单变量区域)
        'IA', 'AS', 'IS', 'GA', 'GI', 'GS',  # 4-9 (双变量区域)
        'IAS', 'GIA', 'GAS', 'GIS',          # 10-13 (三变量区域)
        'GIAS'                               # 14 (四变量区域)
    ]
    
    details_list = []
    for i, key in enumerate(order_keys):
        count = data_dict.get(key, 0)
        rate = count / total
        if rate > 0:
            en = -rate * math.log10(rate)
        else:
            en = 0
            
        # 判断区域类型（单、双、三、四）
        region_type = f"{len(key)}-Var Region"
        
        details_list.append({
            # 'Index': i,
            'Region': key,
            'Type': region_type,
            'Count': count,
            'Rate': rate,
            'Entropy': en
        })
        
    df_details = pd.DataFrame(details_list)
    
    # ================= 2. 构建 T 值表 (df_t_values) =================
    variables = ['A', 'I', 'S', 'G'] # 基础变量
    
    t_values_list = []
    
    # 计算各维度的 T 值 (2维、3维、4维)
    for r in range(2, 5):
        for subset in itertools.combinations(variables, r):
            # 排序变量名以保持一致性 (如 'AI' vs 'IA')
            subset_sorted = sorted(list(subset))
            subset_name = "".join(subset_sorted)
            
            # 使用自定义算法计算 T 值
            t_val = calculate_custom_t_value(subset_sorted, df_details)
            
            t_values_list.append({
                'Dimension': f"{r}D",
                'Combination': subset_name,
                'T_Value': t_val
            })
            
    df_t_values = pd.DataFrame(t_values_list)
    
    # 打印验证信息 (T4 - GIAS)
    t4_row = df_t_values[df_t_values['Combination'] == 'AGIS'] # 注意排序后 G,I,A,S -> A,G,I,S
    # 或者直接查找长度为4的组合
    if not t4_row.empty:
        print(f"Validation T(4D): {t4_row.iloc[0]['T_Value']:.6f}")
    else:
         # 如果组合名顺序不同，直接按变量查找
         print(f"Validation T(GIAS): {calculate_custom_t_value(['A','I','S','G'], df_details):.6f}")

    return df_details, df_t_values

# 数据
sample_data_2003 = {
    'A': 390952, 
    'I': 99031, 
    'S': 19977, 
    'G': 5718,
    'IA': 25682, 
    'AS': 5627, 
    'IS': 3742,
    'GA': 119705, 
    'GI': 478, 
    'GS': 274,
    'IAS': 815, 
    'GIA': 4573, 
    'GAS': 1209, 
    'GIS': 15,
    'GIAS': 97
}

# 运行
if __name__ == "__main__":
    df_det, df_t = run_optimization_analysis(sample_data_2003)
    
df_det
df_t
