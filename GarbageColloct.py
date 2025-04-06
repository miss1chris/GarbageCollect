import gc
import psutil
import numpy as np
import matplotlib.pyplot as plt
import time

# Enable garbage collection
gc.enable()

# Get statistical information of garbage collection
def get_gc_stats():
    return {
        'collected': gc.get_count()[0],  # Number of objects that have been recycled
        'uncollectable': gc.get_count()[1],  # Number of objects that cannot be recycled
        'threshold': gc.get_threshold(),  #Recovery threshold of each generation
    }

# Manually trigger garbage collection
def trigger_gc():
    collected = gc.collect()  # 返回回收的对象数量
    print(f"Garbage collected: {collected} objects")
    return collected

# Get the memory usage of a specific program.
def get_process_memory_usage(process_name):
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        if proc.info['name'] == process_name:
            print(f"Found process '{process_name}' with memory usage: {proc.info['memory_info'].rss} bytes")
            return proc.info['memory_info'].rss  # Resident Set Size
    print(f"Process '{process_name}' not found!")
    return None

# Simulate memory allocation and release (infinite loop)
def simulate_memory_usage():
    data = []
    while True:  # infinite loop
        for i in range(100):
            data.append([0] * 1000000)  # allocate 1MB of memory
            time.sleep(1)
            if i % 10 == 0:
                # deallocating memory
                data.clear()
                print("Memory released")
            print(f"Memory allocated: {len(data)} MB")
            yield len(data)  # Returns the current memory usage (MB)

# Define state space
def get_state(process_name, memory_simulator=None):
    if memory_simulator is not None:
        try:
            # Using a memory emulator
            memory_usage_mb = next(memory_simulator)
        except StopIteration:
            # If the generator runs out, reinitialize it.
            memory_simulator = simulate_memory_usage()
            memory_usage_mb = next(memory_simulator)
    else:
        # Monitoring target program
        memory_usage = get_process_memory_usage(process_name)
        if memory_usage is None:
            raise ValueError(f"Process '{process_name}' not found!")
        memory_usage_mb = memory_usage / (1024 * 1024)
    gc_stats = get_gc_stats()
    return int(min(memory_usage_mb, 1000)), gc_stats['collected']  # Limit the scope of memory_usage.

# Define action space
actions = ['trigger_gc', 'do_nothing']

# Define reward function
def calculate_reward(memory_usage, gc_collected, action):
    if action == 'trigger_gc':
        reward = -memory_usage / 1000 + gc_collected / 100  # Increase sensitivity to memory usage
    else:
        reward = -memory_usage / 1000
    return reward

# Initialize q table
state_size = 1000  # 假设状态空间大小为1000（MB）
action_size = len(actions)
Q = np.random.rand(state_size, action_size) * 0.01  # Initialize the Q table to a smaller random value.

# Q-learning paramters
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 1.0  # Initial exploration rate

# choose actions
def choose_action(state):
    state = min(state, 999)  # Limit state range
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(actions)  # explor
    else:
        return actions[np.argmax(Q[state])]  # using

# 更新Q表
def update_q_table(state, action, reward, next_state):
    state = min(state, 999)  # Limit state range
    next_state = min(next_state, 999)  # Limit next range
    action_index = actions.index(action)
    old_value = Q[state, action_index]
    next_max = np.max(Q[next_state])
    new_value = old_value + alpha * (reward + gamma * next_max - old_value)
    Q[state, action_index] = new_value

# 模拟环境
def simulate_environment(state, action, memory_simulator=None):
    memory_usage, gc_collected = state
    if action == 'trigger_gc':
        collected = min(trigger_gc(), 999)  # limit collected range
        if memory_simulator is not None:
            try:
                next_memory_usage_mb = next(memory_simulator)
            except StopIteration:
                memory_simulator = simulate_memory_usage()
                next_memory_usage_mb = next(memory_simulator)
        else:
            next_memory_usage = get_process_memory_usage(process_name)
            if next_memory_usage is None:
                raise ValueError(f"Process '{process_name}' not found!")
            next_memory_usage_mb = next_memory_usage / (1024 * 1024)
        next_state = int(min(next_memory_usage_mb, 999))  # Use only memory_usage as the state.
        reward = calculate_reward(memory_usage, collected, action)
    else:
        if memory_simulator is not None:
            try:
                next_memory_usage_mb = next(memory_simulator)
            except StopIteration:
                memory_simulator = simulate_memory_usage()
                next_memory_usage_mb = next(memory_simulator)
        else:
            next_memory_usage = get_process_memory_usage(process_name)
            if next_memory_usage is None:
                raise ValueError(f"Process '{process_name}' not found!")
            next_memory_usage_mb = next_memory_usage / (1024 * 1024)
        next_state = int(min(next_memory_usage_mb, 999))  # Use only memory_usage as the state.
        reward = calculate_reward(memory_usage, 0, action)
    return next_state, reward

# training Q-learning
def train_q_learning(process_name,epsilon, episodes=100, steps_per_episode=100, use_memory_simulator=False):
    memory_usages = []
    gc_counts = []
    rewards = []
    iterations = []
    start_time = time.time()
    memory_simulator = simulate_memory_usage() if use_memory_simulator else None
    for episode in range(episodes):
        state = get_state(process_name, memory_simulator)
        episode_rewards = []
        for step in range(steps_per_episode):
            state_value = state[0]  # Use only memory_usage as the state.
            action = choose_action(state_value)
            next_state, reward = simulate_environment(state, action, memory_simulator)
            update_q_table(state_value, action, reward, next_state)
            state = (next_state, 0)  # update state
            episode_rewards.append(reward)
            iterations.append(episode * steps_per_episode + step)
        memory_usages.append(state[0])
        gc_counts.append(get_gc_stats()['collected'])
        rewards.append(np.mean(episode_rewards))  # Record the average reward for each episode.
        epsilon = max(0.01, epsilon * 0.995)  # Dynamic attenuation epsilon
        time.sleep(1)  # Conduct training every 1 second.
    end_time = time.time()
    run_time = end_time - start_time
    return memory_usages, gc_counts, rewards, iterations, run_time

# Baseline experiment without Q-learning，Mainly used for comparative experiments.
def baseline_experiment(process_name, episodes=100, steps_per_episode=100, use_memory_simulator=False):
    memory_usages = []
    gc_counts = []
    rewards = []
    iterations = []
    start_time = time.time()
    memory_simulator = simulate_memory_usage() if use_memory_simulator else None
    for episode in range(episodes):
        state = get_state(process_name, memory_simulator)
        episode_rewards = []
        for step in range(steps_per_episode):
            if memory_simulator is not None:
                try:
                    next_memory_usage_mb = next(memory_simulator)
                except StopIteration:
                    memory_simulator = simulate_memory_usage()
                    next_memory_usage_mb = next(memory_simulator)
            else:
                next_memory_usage = get_process_memory_usage(process_name)
                if next_memory_usage is None:
                    raise ValueError(f"Process '{process_name}' not found!")
                next_memory_usage_mb = next_memory_usage / (1024 * 1024)
            state = (int(min(next_memory_usage_mb, 999)), 0)
            episode_rewards.append(-state[0] / 1000)  # Rewards are based only on memory usage.
            iterations.append(episode * steps_per_episode + step)
        memory_usages.append(state[0])
        gc_counts.append(get_gc_stats()['collected'])
        rewards.append(np.mean(episode_rewards))  # Record the average reward for each episode.
        time.sleep(1)
    end_time = time.time()
    run_time = end_time - start_time
    return memory_usages, gc_counts, rewards, iterations, run_time

# Name of the monitored program
process_name = "chrome.exe"  #Select the edge browser.

# If the memory usage of the target program has not changed, the memory simulator is enabled.
use_memory_simulator = True

# Run the Q-learning experiment
q_memory_usages, q_gc_counts, q_rewards, q_iterations, q_run_time = train_q_learning(process_name, epsilon,use_memory_simulator=use_memory_simulator)

# Running baseline experiment
b_memory_usages, b_gc_counts, b_rewards, b_iterations, b_run_time = baseline_experiment(process_name, use_memory_simulator=use_memory_simulator)

# Draw multiple charts
plt.figure(figsize=(18, 24))

# Chart 1: Comparison of memory usage before and after Q-learning optimization
plt.subplot(6, 1, 1)
plt.plot(q_memory_usages, label='With Q-learning', color='blue')
plt.plot(b_memory_usages, label='Without Q-learning', color='red', linestyle='--')
plt.xlabel('Episode')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage Over Time: With vs Without Q-learning')
plt.legend()

# Chart 2: Comparison of garbage collection times before and after Q-learning optimization
plt.subplot(6, 1, 2)
plt.plot(q_gc_counts, label='With Q-learning', color='blue')
plt.plot(b_gc_counts, label='Without Q-learning', color='red', linestyle='--')
plt.xlabel('Episode')
plt.ylabel('GC Count')
plt.title('Garbage Collection Count Over Time: With vs Without Q-learning')
plt.legend()

# Chart 3: Comparison of reward values before and after Q-learning optimization
plt.subplot(6, 1, 3)
plt.plot(q_rewards, label='With Q-learning', color='blue')
plt.plot(b_rewards, label='Without Q-learning', color='red', linestyle='--')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward Over Time: With vs Without Q-learning')
plt.legend()

# Chart 4: Comparison of iteration times before and after Q-learning optimization
plt.subplot(6, 1, 4)
plt.plot(q_iterations, label='With Q-learning', color='blue')
plt.plot(b_iterations, label='Without Q-learning', color='red', linestyle='--')
plt.xlabel('Step')
plt.ylabel('Iteration')
plt.title('Iteration Over Time: With vs Without Q-learning')
plt.legend()

# Chart 5: Comparison of running time before and after Q-learning optimization
plt.subplot(6, 1, 5)
plt.bar(['With Q-learning', 'Without Q-learning'], [q_run_time, b_run_time], color=['blue', 'red'])
plt.xlabel('Experiment')
plt.ylabel('Run Time (s)')
plt.title('Run Time Comparison: With vs Without Q-learning')

# Chart 6: Comparison of memory usage efficiency before and after Q-learning optimization
plt.subplot(6, 1, 6)
plt.scatter(q_gc_counts, q_memory_usages, label='With Q-learning', color='blue', alpha=0.5)
plt.scatter(b_gc_counts, b_memory_usages, label='Without Q-learning', color='red', alpha=0.5)
plt.xlabel('GC Count')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage vs Garbage Collection Count: With vs Without Q-learning')
plt.legend()

plt.tight_layout()
plt.show()