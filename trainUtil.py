from tf_agents.trajectories import trajectory


def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
    for i in range(steps):
        collect_step(env, policy, buffer)


def compute_avg_return(environment, policy, num_episodes=10):
    print("[compute_avg_return] start, episode num = {0}".format(num_episodes))
    total_return = 0.0
    for e in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        i = 0
        while not time_step.is_last():
            #print("[compute_avg_return] episode: {0} step: {1}".format(e, i))
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            i = i + 1
        total_return += episode_return
    avg_return = total_return / num_episodes
    rtn = avg_return.numpy()[0]
    print("[compute_avg_return] complete, avg_rtn={0}".format(rtn))
    return rtn
