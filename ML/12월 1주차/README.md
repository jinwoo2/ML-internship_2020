
 < OpenAI Gym에서 Chartpole 게임에 DQN을 적용한 튜토리얼과 설명을 확인할 수 있다.>
 
 환경 


![noname01](https://user-images.githubusercontent.com/38103094/101144915-5ff95900-365c-11eb-9749-992ad91547b7.png)


검은색 카트(Cart) 위에 막대기(Pole)를 살짝 올려 놓고
검은색 카트를 좌우로 움직여서 막대기가 쓰러지지 않도록 하는 것이 목표다.


![noname01](https://user-images.githubusercontent.com/38103094/101144968-71dafc00-365c-11eb-831b-88f12ae03b97.png)


먼저 cartpole 환경과 상태의 정보를 먼저 보면
4개의 상태와 2개의 이산적인 액션 으로 구성되어 있다.

위에 Action space에 2개의 액션은  left right이다.
A = {0, 1} 
![noname01](https://user-images.githubusercontent.com/38103094/101145026-8919e980-365c-11eb-8335-bc5e2f571959.png)


그리고 카트의 위치를 나타내는 on_space의 범위를 보면 -48에서 4.8의 범위를 가지는 것을 확인 할 수 있다. 


![noname01](https://user-images.githubusercontent.com/38103094/101145083-9f27aa00-365c-11eb-8d3e-c7205e96cdcc.png)


 지금 보이는 observation 값에 들어간 변수들이 무엇인지를 파악을 해야한다.
 각각  [ 카트의 위치, 카트의 속도, 막대기의 각도, 막대기의 회전율 ]을 의미한다.
지금 env.step(0)을 넣었는데  이는 카트의 속도가 -값이 되었다.  


에러 해결


![noname01](https://user-images.githubusercontent.com/38103094/101145112-ac449900-365c-11eb-98f3-f30e9ebf9daf.png)


계속해서 env.render에 에러가 나와 아나콘다 환경에 gym을 설치하고 주피터를 연결하는 방식으로 해결하고자 했다.

아나콘다에 gym환경 만들고 주피터를 연결시킨다


![noname01](https://user-images.githubusercontent.com/38103094/101145150-b9fa1e80-365c-11eb-86f7-4e9f2243be32.png)


     ---------------------------에러 해결----------------------------
위 코드는 행동 0은 카트에 왼쪽 방향으로 힘을 가하는 것이다. (1은 오른쪽 방향으로 힘을 받는다 ) 막대기가 카트의 진행방향과 반대방향(시계 방향 회전CW)으로 떨어지는 것을 확인 할 수 있다. 보면 막대기의 각도는 3번째 변수를 보면 되는데 -0.03 -> 0.236  대략 0.239 (대략 13도)

<공학에서는 라디안을 많이 써서 라디안으로 그냥 받아들였는데 확실히 알고자 찾아본 결과 라디안이 맞았다.>
행동 1을 줬을 때 오른쪽으로 가고 막대기는 


![noname01](https://user-images.githubusercontent.com/38103094/101145185-c9796780-365c-11eb-8b9a-dc7ea78f1507.png)


여기선 막대기가 카트의 진행방향과 반대방향(반시계 방향 회전CCW)으로 떨어지는 것을 확인 할 수 있다.

여기서 바로 q-network를 짜는 것보다 간단하게 막대기가 기운 방향과 반대로 힘을 가하는 기초적인 알고리즘을 짜 볼 수 있다.

import gym
from gym.wrappers import Monitor

env = gym.make('CartPole-v0')
observation = env.reset()

for i in range(100):
    env.render()
    # 알고리즘1:
    # 막대기가 오른쪽으로 기울어져 있다면, 오른쪽으로 힘을 가하고
    # 그렇지 않다면, 왼쪽으로 힘을 가하기.
    if observation[2] > 0:
        action = 1
    else: action = 0

    observation, reward, done, info = env.step(action)
    print(observation, done)
    if done:
        print(i+1)
        break
env.close()


![noname01](https://user-images.githubusercontent.com/38103094/101145296-edd54400-365c-11eb-8591-73e5ecd34b23.png)

 .i번 횟수를 보면 26번 째에 멈추는 것을 확인할 수 있다.
 
(※지금 if done: 이라고 조건을 주었는데 openai에서 제공하는 cartpole은 막대기의 각도가 12도 이상 기울겄거나 카트가 2.4칸을 움직여서 화면에서 나가게 된다면 떨어졌다고 판단한다고 한다. 그러므로 done이라 함은 막대기가 떨어졌을때라고 생각하면 된다.)


우리는 cart pole 게임에서 막대가 한 쪽으로 기울였을 때 다른 쪽으로 움직이라는 
if observation[2] > 0:
이 구문을 썼는데  강화학습에서 이 구문을 정책(policy) 라고 한다.
지금은 단순 양수 음수만을 가지고 정책을 만들었지만 앞으로 당장의 reward나 누적 reward를 가지는 q값을 가지고 판단을 하는 q-learning에 대해서 만들고자 한다.


먼저 나는 tensorflow로 하나씩 구현하는 것보다 라이브러리로 잘 정리된 keras를 이용해서 뉴럴 네트워크를 구성하고자 하였다.
그렇기에 keras를 먼저 공부하는 시간을 가졌다.


keras.Sequential( [  이 함수는 [] 내부에 layer는 연속적이게 쌓아주는 함수이다.
   layers.dens(64 , activation='relu' , input_shape=(32, ))
   초기 32차원 [배열의 개수 32개]로 받아 relu 활성함수를 거쳐  64차원으로 뱉는 층
 ])
이런식으로 간단하게 구현 할 수 있다.
이에 아래와 같은 예제코드로 층을 여러개 구현할 수 있다.
64개의 유닛을 가진 완전 

model = tf.keras.Sequential([
//  64개의 유닛을 가진 완전 연결 층을 모델에 추가합니다:
layers.Dense(64, activation='relu', input_shape=(32,)),
//  또 하나를 추가합니다:
layers.Dense(64, activation='relu'),
//  10개의 출력 유닛을 가진 소프트맥스 층을 추가합니다:
layers.Dense(10, activation='softmax')])

