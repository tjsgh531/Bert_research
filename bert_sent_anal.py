# 예제1) pipeline을 이용한 간단한 감성분석
"""
from transformers import pipeline

clf = pipeline("sentiment-analysis")
result = clf("what a beautiful day!")[0]
print("감성분석 결과: %s, 감성스코어: %0.4f" %(result['label'], result['score']))
"""

# 예제2) 두 문장의 의미적 유사성 예측
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Auto Classes를 이용한 토크나이저와 모형 자동설정
# mrpc : 의미적으로 유사한 문장의 페어와 그렇지 않은 문장의 페어로 구성 => 두 문장의 의미적 유사성을 학습한 데이터 셋
# 두 문장의 의미적 유사성을 예측하는 모델 만들때 사용
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc") 
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased-finetuned-mrpc"
)

# 의미적으로 유사한 두 문장 선언
input_sentence = "She angered me with her inappropriate comments, rumor-spreading, and disrespecfulness at the formal dinner table"
target_sequence = "She made me angry when she was rude at dinner"

# 토큰화
tokens = tokenizer(input_sentence, target_sequence, return_tensors="pt")

# 모형으로 결과 예측
logits = model(**tokens).logits

# 소프트맥스를 이용해 확률 반환
results = torch.softmax(logits, dim=1).tolist()[0]

for i, label in enumerate(["no", "yes"]):
    print(f"{label}: {int(round(results[i] * 100))}%")