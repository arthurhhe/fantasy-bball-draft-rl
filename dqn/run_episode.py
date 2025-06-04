import numpy as np # pylint: disable=unused-import
import torch # pylint: disable=unused-import
import torch.nn.functional as F

def run_episode(
    env,
    q_net, # pylint: disable=unused-argument
    steps_per_episode,
):
    """Runs the current policy on the given environment.

    Args:
        env (gym): environment to generate the state transition
        q_net (QNetwork): Q-Network used for computing the next action
        steps_per_episode (int): number of steps to run the policy for

    Returns:
        episode_experience (list): list containing the transitions
                        (state, action, reward, next_state)
        episodic_return (float): reward collected during the episode
    """

    # list for recording what happened in the episode
    episode_experience = []
    episodic_return = 0.0

    # reset the environment to get the initial state
    state = env.reset()

    for _ in range(steps_per_episode):
        input_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = q_net(input_tensor)
        probs = F.softmax(q_values, dim=-1)
        num_actions = probs.shape[-1]

        # Availability mask
        available_actions = env._get_obs()
        availability_mask = torch.tensor(available_actions[:num_actions], dtype=torch.float32)
        probs = probs * availability_mask

        # Early bias
        decay = torch.exp(-torch.arange(num_actions, dtype=torch.float32) / 30.0) # decay factor
        probs = probs * decay
        probs = probs / probs.sum()

        action = torch.multinomial(probs, num_samples=1).item()
        next_state, reward, done, _info = env.step(action)

        transition = (state, action, reward, next_state)
        episode_experience.append(transition)

        episodic_return += reward

        state = next_state

        if done:
            # if env._evaluate_team(done=True) >= 0.6: print('Winning Team: ', env._get_rl_team_roster())
            break

    return episode_experience, episodic_return
