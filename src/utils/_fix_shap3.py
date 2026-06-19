import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

path = 'D:/Research_AI_Bio/07_Reports/论文初稿.md'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix old SHAP ranking numbers in section 3.0
# These are from an OLD experiment and don't match our current new_neg results
old_ranking = '在全部656维融合特征（640维ESM-2嵌入 + 16维理化特征）中，前10位核心特征按mean(|SHAP|)排序依次为：ESM-2嵌入维度512（mean|SHAP|=0.082）、Net Charge（0.064）、pI（0.057）、嵌入维度128（0.051）、嵌入维度640（0.048）、嵌入维度256（0.041）、Ser含量（0.039）、嵌入维度64（0.037）、Lys含量（0.035）、嵌入维度384（0.034）。'

new_ranking = '在全部656维融合特征中，前10位核心特征均为ESM-2嵌入维度（mean|SHAP|：ESM_172=0.243、ESM_355=0.234、ESM_512=0.192、ESM_99=0.189、ESM_353=0.182、ESM_214=0.181、ESM_5=0.173、ESM_46=0.150、ESM_349=0.140、ESM_393=0.130）。理化特征中，pos_ratio（mean|SHAP|=0.048）、charge_ratio（0.048）和疏水力矩（0.026）位居前三，其次为neg_ratio（0.021）、分子量（0.016）。'

if old_ranking in content:
    content = content.replace(old_ranking, new_ranking)
    print('Fix 3: updated SHAP ranking numbers')
else:
    print('Fix 3: old ranking text not found')
    idx = content.find('656维融合特征')
    if idx >= 0:
        print('Partial match:', repr(content[idx:idx+200]))

# Also update the Ser/Lys sentence that references the old ranking
old_ser = '**理化特征提供互补增益**：Ser含量（第7位，mean|SHAP|=0.039）和Lys含量（第9位，0.035）进入前10位，反映特定氨基酸组成对AMP活性的贡献——Lys（赖氨酸）提供正电荷，Ser（丝氨酸）参与氢键网络和膜界面识别。这与特征消融实验中融合特征F1（0.962）显著优于仅ESM-2（0.946）的结论一致（表6），理化特征虽贡献不及ESM-2嵌入，但在区分ESM-2嵌入相似但功能不同的序列时提供了关键的互补信息。'

new_ser = '**理化特征提供互补增益**：在16项理化特征中，正电荷比（pos_ratio, mean|SHAP|=0.048）、电荷比（charge_ratio, 0.048）和疏水力矩（hmoment, 0.026）位居前三。净电荷（charge, 0.011）和疏水性（hydrophob, 0.011）提供稳定但量级较小的贡献。这与特征消融实验中融合特征F1（0.962）显著优于仅ESM-2（0.946）的结论一致（表6）——理化特征虽贡献不及ESM-2嵌入，但在区分ESM-2嵌入相似但功能不同的序列时提供了关键的互补信息。'

if old_ser in content:
    content = content.replace(old_ser, new_ser)
    print('Fix 4: updated Ser/Lys text with correct numbers')
else:
    print('Fix 4: old Ser/Lys text not found')
    # Try partial
    idx = content.find('理化特征提供互补增益')
    if idx >= 0:
        print('Partial:', repr(content[idx:idx+200]))

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('\nDone')
