import jieba

print("/ ".join(jieba.cut("我们中出了一个叛徒", HMM=False)))
