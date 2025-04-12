# 원핫 인코딩에서는 단어 사이의 관계도 표현할 수 없어
# 코사인 유사도를 계산하면 유사도가 0으로 나온다.
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

word_dict = {
    "school": np.array([1, 0, 0]),
    "study": np.array([0, 1, 0]),
    "workout": np.array([0, 0, 1])
}

# 두 단어 사이의 코사인 유사도 계산
cosine_school_study = cosine_similarity([word_dict["school"]], [word_dict["study"]])
cosine_school_workout = cosine_similarity([word_dict["school"]], [word_dict["workout"]])

print(f"cosine_school_study: {cosine_school_study}")
print(f"cosine_school_workout: {cosine_school_workout}")