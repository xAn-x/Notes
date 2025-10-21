
![[Pasted image 20250626191947.png]]

![[Pasted image 20250626191825.png]]

## *Mar*kov Decision Process

A Markov Decision Process (MDP) is a mathematical framework for modelling decision-making in situations where outcomes are partly random and partly under the control of a decision maker. It's defined by:

- **States (S):** A set of possible situations the system can be in.
- **Actions (A):** A set of actions the agent can take in each state.
- **Transition Probabilities ():** The probability of transitioning from state to state after taking action . _This is the "Markov" property – the next state depends only on the current state and action, not the history_.
- **Rewards (R):** A function that assigns a numerical reward to each state-action pair (), or to each state transition (). The goal is to maximize cumulative reward.
- **Discount Factor ():** A value between 0 and 1 that determines the importance of future rewards. A lower  emphasizes immediate rewards.

>[!Summary]
>>An agent in a state chooses an action, receives a reward, and transitions to a new state probabilistically. The goal is to find a policy (a mapping from states to actions) that maximizes the expected cumulative reward.


**Find Policy to maximize return**
$$
\begin{gather}
Policy: \\
\\
\pi: State(S) => action(a) \\
other way \\
\pi(a|S)=probability
\end{gather}
$$


$$
\begin{gather}
\text{Return:} \\ \\
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \\
\text{discount-factor: } 0 \le \gamma \le 1
\end{gather}
$$

### Value Function

![[Pasted image 20250626194858.png]]

A value function estimates how "good" it is for an agent to be in a particular state or take a particular action in a particular state. It's a crucial component because it guides the agent's learning process. There are two main types:

- **State-value function V(s):** Estimates the expected cumulative reward an agent will receive starting from state  and following a given policy . It answers the question: "How good is it to be in state ?"
    
- **Action-value function Q(S,a):** Estimates the expected cumulative reward an agent will receive starting from state , taking action , and then following a given policy . It answers the question: "How good is it to take action  in state ?"


![[Pasted image 20250626195341.png]]