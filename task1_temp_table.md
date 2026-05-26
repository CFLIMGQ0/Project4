# Task 1 显著性表格 LaTeX 代码暂存

下面暂存从 `May17th_Sage_LaTeX_Guidelines.tex` 实验部分移出的两张显著性分析表格。

```latex
\begin{table*}[h]
\scriptsize\sf\centering
\caption{Paired bootstrap significance analysis for the overall model comparison in Table~\ref{T2}. Differences are computed as Label graph MIL minus the comparator model using 2,000 examination-level bootstrap resamples.\label{T7}}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l*{6}{c}}
\toprule
Comparator model & \shortstack{$\Delta$ Macro F1\\(95\% CI)} & \shortstack{Macro F1\\$p$-value} & \shortstack{$\Delta$ Micro F1\\(95\% CI)} & \shortstack{Micro F1\\$p$-value} & \shortstack{$\Delta$ Kappa\\(95\% CI)} & \shortstack{Kappa\\$p$-value} \\
\midrule
Attention MIL & 0.0146 [0.0003, 0.0284] & 0.0430 & 0.0147 [0.0000, 0.0293] & 0.0510 & 0.0312 [0.0010, 0.0612] & 0.0450 \\
Mean pooling & 0.0065 [-0.0076, 0.0192] & 0.3598 & 0.0060 [-0.0080, 0.0193] & 0.3918 & 0.0170 [-0.0120, 0.0447] & 0.2579 \\
Transformer-context MIL & 0.0060 [-0.0079, 0.0201] & 0.4138 & 0.0061 [-0.0085, 0.0212] & 0.4268 & 0.0149 [-0.0151, 0.0452] & 0.3508 \\
Top-$k$ MIL & 0.0044 [-0.0081, 0.0171] & 0.4758 & 0.0051 [-0.0077, 0.0185] & 0.4278 & 0.0169 [-0.0096, 0.0445] & 0.2079 \\
Max pooling & 0.1119 [0.0920, 0.1307] & $<0.001$ & 0.1218 [0.1012, 0.1414] & $<0.001$ & 0.3673 [0.3240, 0.4091] & $<0.001$ \\
TransMIL & 0.0044 [-0.0112, 0.0205] & 0.5887 & 0.0038 [-0.0123, 0.0207] & 0.6487 & 0.0059 [-0.0266, 0.0396] & 0.7076 \\
DSMIL & 0.0049 [-0.0101, 0.0187] & 0.5057 & 0.0046 [-0.0106, 0.0188] & 0.5367 & 0.0104 [-0.0209, 0.0404] & 0.5197 \\
DTFD-MIL & 0.0198 [0.0051, 0.0342] & 0.0130 & 0.0211 [0.0058, 0.0360] & 0.0110 & 0.0623 [0.0303, 0.0944] & $<0.001$ \\
CLAM-MB & 0.0290 [0.0115, 0.0462] & $<0.001$ & 0.0294 [0.0116, 0.0470] & $<0.001$ & 0.0710 [0.0338, 0.1074] & $<0.001$ \\
CLAM-SB & 0.0366 [0.0197, 0.0527] & $<0.001$ & 0.0380 [0.0215, 0.0542] & $<0.001$ & 0.1003 [0.0644, 0.1346] & $<0.001$ \\
\bottomrule
\end{tabular}
}
\end{table*}

\begin{table*}[h]
\scriptsize\sf\centering
\caption{Paired bootstrap significance analysis for the label relation reasoner ablation in Table~\ref{T4}. Differences are computed as Full Label Graph minus the comparator module using 2,000 examination-level bootstrap resamples.\label{T8}}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l*{6}{c}}
\toprule
Comparator module & \shortstack{$\Delta$ Macro F1\\(95\% CI)} & \shortstack{Macro F1\\$p$-value} & \shortstack{$\Delta$ Micro F1\\(95\% CI)} & \shortstack{Micro F1\\$p$-value} & \shortstack{$\Delta$ Kappa\\(95\% CI)} & \shortstack{Kappa\\$p$-value} \\
\midrule
w/o Label Graph Reasoner & 0.0111 [-0.0014, 0.0226] & 0.0740 & 0.0133 [0.0003, 0.0251] & 0.0460 & 0.0439 [0.0166, 0.0694] & 0.0030 \\
Label Self-Attention Reasoner & 0.0090 [-0.0044, 0.0236] & 0.1889 & 0.0091 [-0.0047, 0.0240] & 0.1919 & 0.0229 [-0.0058, 0.0541] & 0.1149 \\
Label Transformer Reasoner & 0.0096 [-0.0059, 0.0244] & 0.2489 & 0.0098 [-0.0058, 0.0251] & 0.2449 & 0.0197 [-0.0117, 0.0519] & 0.2469 \\
Dynamic Label GAT Reasoner & 0.0014 [-0.0113, 0.0138] & 0.8106 & 0.0019 [-0.0111, 0.0147] & 0.7566 & 0.0111 [-0.0165, 0.0378] & 0.4188 \\
Cosine Dynamic Graph Reasoner & 0.0014 [-0.0117, 0.0151] & 0.8786 & 0.0005 [-0.0126, 0.0142] & 0.9795 & 0.0033 [-0.0233, 0.0312] & 0.8746 \\
Static Co-occurrence GCN Reasoner & 0.0021 [-0.0117, 0.0154] & 0.7886 & 0.0012 [-0.0128, 0.0148] & 0.8786 & 0.0045 [-0.0234, 0.0334] & 0.8026 \\
Low-Rank Label Graph Reasoner & 0.0144 [-0.0001, 0.0282] & 0.0530 & 0.0132 [-0.0017, 0.0275] & 0.0810 & 0.0245 [-0.0057, 0.0535] & 0.1109 \\
Label MLP-Mixer Reasoner & 0.0030 [-0.0117, 0.0167] & 0.6997 & 0.0028 [-0.0122, 0.0170] & 0.7326 & 0.0058 [-0.0255, 0.0347] & 0.7436 \\
\bottomrule
\end{tabular}
}
\end{table*}
```
