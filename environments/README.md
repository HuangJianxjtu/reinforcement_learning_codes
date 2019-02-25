How to use this environment?
    1.Install gym
    
        NOTE:don't use pip install else you will be unable to change the environment. So please build from the source files.The fellowing commands shows how to do it:
        
        $git clone https://github.com/openai/gym
        
        $cd gym
        
        $pip install -e .
        
        reference: https://gym.openai.com/docs/
        
    2.Register this environment
    
        2.1 Copy the "grid_maze.py" to this directory: "your gym installation path"+/gym/envs/classic_control
        
            For example, my path is: /home/jian/gym/gym/envs/classic_control
            
        2.2 Add the following commands to the end of "__init__.py", which is in the same directory in 2.1
        
            from gym.envs.classic_control.grid_maze import GridMazeEnv
            
        2.3 Add the following commands to the end of "__init__.py", which is in the upper directory of that in 2.1
        
            register(
            
                id='GridMazeWorld-v0',
                
                entry_point='gym.envs.classic_control:GridMazeEnv',
                
                max_episode_steps=200,
                
                reward_threshold=100.0,
                
                )
                

After finishing the above steps, you can use the environment now! "test_gridMaze.py" is a simple demo that can tell you how to use this environment. "maze.jpg" shows what the maze looks like. "maze_state.jpeg" shows the states in the maze model.
