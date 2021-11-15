# 태권도 동작 데이터

### 1. Usage

>  [guide for NSML](https://n-clair.github.io/ai-docs/_build/html/ko_KR/index.html)

#### How to run

```
nsml run -d TAEKWONDO
```

#### How to check session logs
```
nsml logs -f [SESSION_NAME] # e.g., nsml logs -f teamname/TAEKWONDO/1
```

#### How to list checkpoints saved
You can search model checkpoints by using the following command:
```
nsml model ls teamname/TAEKWONDO/1
```

#### How to submit
The following command is an example of running the evaluation code using the model checkpoint at 10th epoch.
```
nsml submit  teamname/TAEKWONDO/1 1
```

#### How to check leaderboard
```
nsml dataset board teamname
```

### 2. Dataset Info



#### Label Info:
```json
{
    "기본준비": 0,
    "뒷굽이하고 바깥막기": 1,
    "뒷굽이하고 손날바깥막기": 2,
    "앞굽이하고 당겨지르기": 3,
    "앞굽이하고 바탕손안막고 지르기": 4,
    "앞서고 지르기": 5,
    "앞차고 뒷굽이하고 바깥막기": 6,
    "앞차고 앞굽이하고 지르기": 7,
    "앞차고 앞서고 아래막고 지르기": 8, 
    "옆서고 메주먹내려치기": 9
}
```



#### Data Description:

```
\_train
    \_ train_data (folder)
        \_ 00001-S01.jpg (태권도 동작 시작 사진)
        \_ 00001-M01.jpg (태권도 동작 중간 사진)
        \_ 00001-E01.jpg (태권도 동작 끝 사진)
    \_ train_label (imageset_name, label) ex. 00001 001

# of training images: 7777
# of test images: 1111

```

<font color='red'> 해당 label에는 keypoint 정보는 포함되어있지 않습니다.</font>

#### Evaluation Metric:

![image](https://user-images.githubusercontent.com/59240255/141822773-800febee-9b1c-4d2f-938b-f514871054f0.png)


