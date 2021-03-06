
... 
이번 주에 DQN을 공부하고 있던 도중 Reinforce 알고리즘을 먼저 알아야 DQN을 짤 수 있다는 것을 알게 됬다.

무슨 말이냐면 REINFORCE알고리즘(한 에피소드 마다 학습)의 발전된 형태로 Actor-Critic( 타임스텝마다 학습이 이루어짐)라서 그렇다.


우선 DQN을 공부했을때 딥살사에서 마찬가지로 오차함수를 MSE[(정답- 예측)^2]를 써서 오차함수를 최소화 하는 방향으로 인공신경망이 업데이트
된다는 것은 파악을 하였다.

그리고 replay 메모리방식으로 에이전트가 학습을 해서 점점 높은 점수를 받게 되면 더 좋은 샘플들이 리플레이 메모리에 저장된다. 

딥살사에서 on-policy방식에서의 단점인 '안 좋은 상황에 빠려버리면 탈출할 수 없는 상황이 존재한다' 를 DQN에서는 해결할 수 있다.

매 타임스텝마다 인공신경망을 업데이트 한다면, 안 좋은 상황에서 계속적으로 그 안 좋은 상황에 갖혀 벗어날 수 없다는 말이다.

그렇기에 임의의 공간에 여러 sample들을 저장해 놓고 무작위로 sample 추출을 하여 서로 시간적인 상관관계를 없엘 수 있기에 가능하다는 말
인 것 같다.

1. 현재 정책으로부터 발생된 상황(에이전트의 선택으로 발생된 상태)
2. 이전의 정책으로 발생한 상황(이전의 선택들로부터 발생한 상태)



![캡처](https://user-images.githubusercontent.com/38103094/103468340-78fc4380-4d9b-11eb-9b6c-d1b40eaddb3f.PNG)

#1 번째

오픈AI에서 CartPole-v1은 500번의 타임스탭으로 한정시킨 cartpole 버전이다

위에 스크린샷을 보면 env객체로 선언하고 그 객체로 환경 정보를 가저온 것이다.

![캡처](https://user-images.githubusercontent.com/38103094/103468362-b5c83a80-4d9b-11eb-9a74-ed7520eb4944.PNG)

#2 번째

이 부분은 에이전트가 인공신경망을 생성하는데 필요한 상태와 행동의 크기를 불러 온 것이다.

상태와 행동의 크기를 환경으로부터 얻었으면 그 정보를 통해 다음 에이전트 객체를 생성한다.

![캡처](https://user-images.githubusercontent.com/38103094/103468393-f758e580-4d9b-11eb-8309-295ab2205c57.PNG)

이제 DQNAgent 클래스에서 어떤 함수가 필요한지, 환경과의 어떤 상호작용을 하는지 먼저 찾아봐야 한다. 

###1. 상태에 따른 행동 선택

###2. 선택한 행동으로 환경에서 한 타임스텝을 진행

###3. 환경으로부터 다음 상태와 보상을 받음

###4. 샘플을 리플레이 메모리에 저장

###5. 리플레이 메모리에서 무작위 추출한 샘플을 학습

###6. 에피소드마다 타킷 모델 업데이트

대락 6가지의 기능을 담고 있어야 한다.

모델을 생성하는 build_model을 보자

![캡처](https://user-images.githubusercontent.com/38103094/103468456-6c2c1f80-4d9c-11eb-996f-5457615f67d7.PNG)

DQN알고리즘에서 특징이 '타겟신경망'을 사용한다는 것이다. 

build_model함수를 2번 호출해서 model과 target_model을 생성한다.
이 2개의 model은 가중치가 무작위로 초기화 되기 때문에 두 모델은 같지 않다. 따라서 학습을 시작하기 전에는 두 
모델의 가중치 값을 통일시켜야 한다. 그래서 그 역할 해주는 것이 update_target_model 함수이다.

![캡처](https://user-images.githubusercontent.com/38103094/103468563-9500e480-4d9d-11eb-8710-534633b48c47.PNG)

update_target_model은 학습을 진행하는 도중에도 target_model과 model로 업데이트하는 역할을 한다. 


![캡처](https://user-images.githubusercontent.com/38103094/103468729-a1863c80-4d9f-11eb-96f0-13b440e27d67.PNG)



ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

Actor 네트워크와 Critic 네트워크라는 두 개의 네트워크를 사용합니다.

Actor는 상태가 주어졌을 때 행동을 결정하고, Critic은 상태의 가치를 평가합니다.

 Actor-Critic 은 Replay Buffer 를 사용하지 않고, 매 step 마다 얻어진 상태(s), 행동(a), 보상(r), 다음 상태(s’)를 이용해서 모델을 학습시킵니다.
