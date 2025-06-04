from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import torch
from game.reward_values import REWARD_WIN_BUFFER_THRESHOLD

def train_agent(env, agent, buffer, win_buffer, episodes=10000, batch_size=128, warmup_episodes=2000, updates_per=20, update_every=50, log_dir='runs'):
    start_time = time.time()
    episode_rewards = []
    writer = SummaryWriter(log_dir=f"{log_dir}/ac/{time.strftime('%Y%m%d-%H%M%S')}")
    global_step = 0
    episode = 0

    obs = env.reset()
    episode_reward = 0
    # episode_transitions = []
    print(f"Max Steps: {episodes * env.roster_size + 1}")
    for step in range(1, episodes * env.roster_size + 1):
        action = agent.act(obs, eval_mode=False)
        next_obs, reward, done, _ = env.step(action)
        
        # episode_transitions.append((obs, action, reward, next_obs, done))
        if reward > 0:
            win_buffer.add(obs, action, reward, next_obs, done)

        buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs
        episode_reward += reward

        if done:
            obs = env.reset()
            writer.add_scalar("Train/EpisodeReward", episode_reward, episode)
            episode_rewards.append(episode_reward)
            
            # if (episode_reward >= REWARD_WIN_BUFFER_THRESHOLD):
            #     for transition in episode_transitions:
            #         win_buffer.add(*transition)

            episode_reward = 0
            # episode_transitions = []
            episode += 1

            if episode > warmup_episodes and episode % update_every == 0:
                for _ in range(updates_per):
                    half_batch = batch_size // 2
                    if (len(win_buffer) < half_batch):
                        critic_batch = buffer.sample(batch_size)
                        actor_batch = buffer.sample(batch_size)
                    else:
                        critic_batch_main = buffer.sample(half_batch)
                        actor_batch_main = buffer.sample(half_batch)
                        critic_batch_win = win_buffer.sample(half_batch)
                        actor_batch_win = win_buffer.sample(half_batch)
                        critic_batch = tuple(torch.cat([a, b], dim=0) for a, b in zip(critic_batch_main, critic_batch_win))
                        actor_batch = tuple(torch.cat([a, b], dim=0) for a, b in zip(actor_batch_main, actor_batch_win))

                    try:
                        critic_metrics = agent.update_critic(critic_batch)
                        actor_metrics = agent.update_actor(actor_batch)
                    except StopIteration:
                        print("Warning: StopIteration during update. Skipping this step.")
                        continue

                    # Log losses
                    if critic_metrics:
                        for k, v in critic_metrics.items():
                            writer.add_scalar(f"Losses/{k}", v, global_step)
                    if actor_metrics:
                        for k, v in actor_metrics.items():
                            writer.add_scalar(f"Losses/{k}", v, global_step)

        if episode % 100 == 0 and step % env.roster_size == 0:
            print(f"Episodes: {episode}, Buffer Size: {len(buffer)}, Win Buffer Size: {len(win_buffer)}, Step: {step}, Time Elapsed: {time.time() - start_time}")
        if episode % 100 == 0  and step % env.roster_size == 0:
            eval_reward = evaluate_agent(env, agent)
            writer.add_scalar("Eval/AvgReward", eval_reward, step)
            print(f"[Episodes: {episode}] Eval Reward: {eval_reward:.2f}")

        global_step += 1

    writer.close()
    return episode_rewards

def evaluate_agent(env, agent, episodes=10):
    total_reward = 0
    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action = agent.act(obs, eval_mode=True)
            next_obs, reward, done, _ = env.step(action)
            total_reward += reward
            obs = next_obs
            assert not np.isnan(reward), "NaN reward"
            assert not np.any(np.isnan(obs)), "NaN in observation"
    return total_reward / episodes
